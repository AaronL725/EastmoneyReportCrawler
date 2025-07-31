#!/usr/bin/env python3
"""
PDF批量处理工具 - 完整版本
功能：将PDF文件转换为包含图像描述的Markdown文件
作者：Assistant能：将PDF文件转换为包含图像描述的Ma
日期：2025-07-27

主要功能：
1. 批量处理PDF文件，提取文本和图像
2. 使用Ollama模型为图像生成中文描述
3. 输出结构化的Markdown文件
4. 支持断点续传和进度跟踪
5. 自动内存管理和清理
"""

import copy
import json
import os
import re
import base64
import requests
import shutil
import gc
import time
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Set
from datetime import datetime

from loguru import logger

# MinerU相关导入
from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.enum_class import MakeMode
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path


class OllamaImageDescriber:
    """使用Ollama API描述图片内容的类"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5vl:3b-q4_K_M"):
        self.base_url = base_url
        self.model = model
        self.api_url = f"{base_url}/api/generate"
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """将图片编码为base64格式"""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded_string
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return None
    
    def describe_image(self, image_path: str, prompt: str = None) -> str:
        """使用Ollama模型描述图片内容"""
        
        # 默认的高级提示词 - 专门针对金融工程研究报告图片设计
        if prompt is None:
            prompt = """你是一位专业的金融图表分析师。请仔细观察这张图片，用中文详细描述其内容。请严格按照客观事实描述，不要进行主观解读。

请按以下格式描述：

【图表类型】：明确说明这是什么类型的图表（如：折线图、柱状图、K线图、散点图、表格等）

【标题信息】：如果有标题，请完整准确地摘录

【坐标轴】：
- X轴：标签内容、数值范围、刻度间隔
- Y轴：标签内容、数值范围、刻度间隔
- 如有右Y轴，也要描述

【数据系列】：描述图中有几条线/柱子/数据系列，每个系列的颜色、样式、图例名称

【关键数值】：指出图中的最高点、最低点、重要转折点的具体数值和位置

【图例和标注】：描述图例位置、内容，以及图中的文字标注、箭头等

【其他元素】：网格线、背景色、水印等视觉元素

请用具体的数字和描述词填充以上内容，不要使用占位符或方括号。"""
        
        try:
            # 编码图片
            base64_image = self.encode_image_to_base64(image_path)
            if not base64_image:
                return f"[图片加载失败: {os.path.basename(image_path)}]"
            
            # 构建请求数据
            data = {
                "model": self.model,
                "prompt": prompt,
                "images": [base64_image],
                "stream": False
            }
            
            # 发送请求，增加重试机制
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.post(self.api_url, json=data, timeout=120)
                    response.raise_for_status()
                    break
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        raise e
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in 5 seconds...")
                    time.sleep(5)
            
            # 解析响应
            result = response.json()
            description = result.get("response", "").strip()
            
            if description:
                logger.info(f"Successfully described image: {os.path.basename(image_path)}")
                return description
            else:
                return f"[无法生成图片描述: {os.path.basename(image_path)}]"
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed for {image_path}: {e}")
            return f"[图片描述服务异常: {os.path.basename(image_path)}]"
        except Exception as e:
            logger.error(f"Failed to describe image {image_path}: {e}")
            return f"[图片描述失败: {os.path.basename(image_path)}]"


class ProcessingTracker:
    """处理进度跟踪器，支持断点续传和避免重复处理"""
    
    def __init__(self, progress_file: str):
        self.progress_file = progress_file
        self.processed_files: Set[str] = set()
        self.failed_files: Set[str] = set()
        self.load_progress()
    
    def load_progress(self):
        """从文件加载已处理的文件记录"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.processed_files = set(data.get('processed', []))
                    self.failed_files = set(data.get('failed', []))
                logger.info(f"Loaded progress: {len(self.processed_files)} processed, {len(self.failed_files)} failed")
            except Exception as e:
                logger.error(f"Failed to load progress file: {e}")
    
    def save_progress(self):
        """保存处理进度到文件"""
        try:
            data = {
                'processed': list(self.processed_files),
                'failed': list(self.failed_files),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def is_processed(self, file_hash: str) -> bool:
        """检查文件是否已处理"""
        return file_hash in self.processed_files
    
    def mark_processed(self, file_hash: str):
        """标记文件为已处理"""
        self.processed_files.add(file_hash)
        self.failed_files.discard(file_hash)  # 如果之前失败过，现在移除失败记录
    
    def mark_failed(self, file_hash: str):
        """标记文件处理失败"""
        self.failed_files.add(file_hash)
    
    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return {
            'processed': len(self.processed_files),
            'failed': len(self.failed_files),
            'total_attempted': len(self.processed_files) + len(self.failed_files)
        }


class PDFBatchProcessor:
    """PDF批量处理器主类"""
    
    def __init__(self, input_folder: str = "Report", output_folder: str = "mdReport", 
                 ollama_base_url: str = "http://localhost:11434", 
                 ollama_model: str = "qwen2.5vl:3b-q4_K_M"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        
        # 创建输出目录
        Path(self.output_folder).mkdir(exist_ok=True)
        
        # 初始化组件
        self.describer = OllamaImageDescriber(ollama_base_url, ollama_model)
        self.tracker = ProcessingTracker(str(Path(self.output_folder) / "processing_progress.json"))
    
    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """计算文件的MD5哈希值，用于唯一标识"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                # 读取文件的前1MB和后1KB来计算哈希，平衡速度和唯一性
                chunk = f.read(1024 * 1024)  # 前1MB
                if chunk:
                    hash_md5.update(chunk)
                
                # 如果文件大于1MB，读取最后1KB
                f.seek(-1024, 2)
                chunk = f.read(1024)
                if chunk:
                    hash_md5.update(chunk)
            
            # 添加文件大小和修改时间以增强唯一性
            stat = os.stat(file_path)
            hash_md5.update(str(stat.st_size).encode())
            hash_md5.update(str(stat.st_mtime).encode())
            
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return f"error_{os.path.basename(file_path)}_{int(time.time())}"
    
    @staticmethod
    def find_all_pdf_files(root_dir: str) -> List[Path]:
        """递归查找所有PDF文件"""
        pdf_files = []
        root_path = Path(root_dir)
        
        if not root_path.exists():
            logger.error(f"Directory not found: {root_dir}")
            return pdf_files
        
        logger.info(f"Scanning for PDF files in: {root_dir}")
        
        for pdf_file in root_path.rglob("*.pdf"):
            if pdf_file.is_file():
                pdf_files.append(pdf_file)
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        return pdf_files
    
    @staticmethod
    def cleanup_memory_and_cache():
        """清理内存和缓存，避免内存爆炸"""
        try:
            # 强制垃圾回收
            gc.collect()
            
            # 清理CUDA缓存（如果有GPU）
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("Cleared CUDA cache")
            except ImportError:
                pass
            
            logger.info("Memory cleanup completed")
        except Exception as e:
            logger.warning(f"Failed to cleanup memory: {e}")
    
    @staticmethod
    def remove_directory_safely(dir_path: str):
        """安全地删除目录"""
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                logger.info(f"Removed directory: {dir_path}")
            else:
                logger.debug(f"Directory not found: {dir_path}")
        except Exception as e:
            logger.error(f"Failed to remove directory {dir_path}: {e}")
    
    def process_markdown_with_image_descriptions(self, md_content: str, image_dir: str) -> str:
        """处理markdown内容，将图片替换为文字描述"""
        
        def replace_image_with_description(match):
            image_path = match.group(1)
            # 如果路径以"images/"开头，则移除它，因为image_dir已经是images目录了
            if image_path.startswith("images/"):
                image_filename = image_path[7:]  # 移除"images/"前缀
            else:
                image_filename = image_path
            
            full_image_path = os.path.join(image_dir, image_filename)
            
            if os.path.exists(full_image_path):
                # 使用ollama模型描述图片
                description = self.describer.describe_image(full_image_path)
                # 格式化描述文本
                return f"\n**[图片描述]**: {description}\n"
            else:
                logger.warning(f"Image file not found: {full_image_path}")
                return f"\n**[图片未找到]**: {image_path}\n"
        
        # 匹配markdown中的图片语法 ![alt](path)
        image_pattern = r'!\[.*?\]\((.*?)\)'
        processed_content = re.sub(image_pattern, replace_image_with_description, md_content)
        
        return processed_content
    
    def do_parse_single_pdf(self, pdf_path: str, output_dir: str) -> str:
        """
        处理单个PDF文件
        Args:
            pdf_path: PDF文件路径
            output_dir: 输出目录
        Returns:
            生成的markdown内容
        """
        file_name = str(Path(pdf_path).stem)
        pdf_bytes = read_fn(pdf_path)
        
        # 使用pipeline模式处理
        new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, 0, None)
        
        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
            [new_pdf_bytes], ['ch'], parse_method="auto", formula_enable=True, table_enable=True
        )
        
        model_list = infer_results[0]
        local_image_dir, local_md_dir = prepare_env(output_dir, file_name, "auto")
        image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
        
        images_list = all_image_lists[0]
        pdf_doc = all_pdf_docs[0]
        _lang = lang_list[0]
        _ocr_enable = ocr_enabled_list[0]
        
        middle_json = pipeline_result_to_middle_json(
            model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable, True
        )
        
        pdf_info = middle_json["pdf_info"]
        
        # 生成markdown内容
        image_dir = str(os.path.basename(local_image_dir))
        md_content_str = pipeline_union_make(pdf_info, MakeMode.MM_MD, image_dir)
        
        # 如果有图像目录，处理图像描述
        if os.path.exists(local_image_dir):
            try:
                md_content_str = self.process_markdown_with_image_descriptions(
                    md_content_str, local_image_dir
                )
                # 清理图像文件夹
                self.remove_directory_safely(local_image_dir)
                logger.info(f"Processed images with Ollama and cleaned up {local_image_dir}")
            except Exception as e:
                logger.error(f"Failed to process images with Ollama: {e}")
        
        return md_content_str
    
    def process_batch(self, resume: bool = True) -> int:
        """
        批量处理PDF文件
        Args:
            resume: 是否恢复之前的进度
        Returns:
            成功处理的文件数量
        """
        if not resume:
            # 清空进度记录
            self.tracker.processed_files.clear()
            self.tracker.failed_files.clear()
        
        # 查找所有PDF文件
        pdf_files = self.find_all_pdf_files(self.input_folder)
        logger.info(f"Found {len(pdf_files)} PDF files in {self.input_folder}")
        
        if not pdf_files:
            logger.warning("No PDF files found to process")
            return 0
        
        # 统计信息
        total_files = len(pdf_files)
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        logger.info(f"Starting batch processing of {total_files} PDF files...")
        
        for i, pdf_file in enumerate(pdf_files, 1):
            try:
                # 计算文件哈希
                file_hash = self.get_file_hash(str(pdf_file))
                
                # 检查是否已经处理过
                if self.tracker.is_processed(file_hash):
                    skipped_count += 1
                    logger.info(f"[{i}/{total_files}] Skipping already processed: {pdf_file.name}")
                    continue
                
                # 生成输出文件路径 - 直接输出到mdReport根目录
                output_md_path = Path(self.output_folder) / f"{pdf_file.stem}.md"
                
                logger.info(f"[{i}/{total_files}] Processing: {pdf_file.name}")
                
                # 创建临时输出目录
                temp_output_dir = output_md_path.parent / f"temp_{pdf_file.stem}"
                temp_output_dir.mkdir(exist_ok=True)
                
                # 处理PDF文件
                md_content = self.do_parse_single_pdf(str(pdf_file), str(temp_output_dir))
                
                # 保存最终的Markdown文件
                with open(output_md_path, 'w', encoding='utf-8') as f:
                    f.write(md_content)
                
                logger.info(f"Saved final markdown to: {output_md_path}")
                
                # 清理临时目录
                self.remove_directory_safely(str(temp_output_dir))
                
                # 更新进度
                self.tracker.mark_processed(file_hash)
                processed_count += 1
                
                # 清理内存
                self.cleanup_memory_and_cache()
                
                logger.info(f"[{i}/{total_files}] Successfully processed: {pdf_file.name}")
                
                # 每处理1个文件保存一次进度
                self.tracker.save_progress()
                logger.info(f"Progress saved. Processed: {processed_count}, Skipped: {skipped_count}, Errors: {error_count}")
                
            except Exception as e:
                error_count += 1
                logger.error(f"[{i}/{total_files}] Failed to process {pdf_file.name}: {e}")
                
                # 标记为失败并继续处理下一个文件
                file_hash = self.get_file_hash(str(pdf_file))
                self.tracker.mark_failed(file_hash)
                continue
        
        # 最终保存进度
        self.tracker.save_progress()
        
        # 打印最终统计
        logger.info(f"""
Batch processing completed!
Total files: {total_files}
Successfully processed: {processed_count}
Skipped (already processed): {skipped_count}
Errors: {error_count}
Output directory: {Path(self.output_folder).absolute()}
""")
        
        return processed_count


def main():
    """主函数"""
    
    # 设置环境变量
    os.environ['MINERU_MODEL_SOURCE'] = "local"
    
    print("PDF批量解析工具")
    print("=" * 50)
    
    # 设置输入和输出目录
    input_folder = "Report"
    output_folder = "mdReport"
    
    # 检查输入目录是否存在
    if not os.path.exists(input_folder):
        logger.error(f"❌ 输入目录不存在: {input_folder}")
        logger.info("请确保Report目录存在并包含PDF文件")
        return
    
    # 初始化处理器
    processor = PDFBatchProcessor(input_folder, output_folder)
    
    # 检查PDF文件数量
    logger.info("🔍 正在扫描PDF文件...")
    pdf_files = processor.find_all_pdf_files(input_folder)
    total_files = len(pdf_files)
    
    if total_files == 0:
        logger.warning(f"⚠️  在 {input_folder} 中未找到PDF文件")
        return
    
    print(f"\n📊 文件统计:")
    print(f"   发现PDF文件: {total_files} 个")
    
    # 显示文件列表
    print(f"\n📋 文件列表 (前10个):")
    for i, pdf_file in enumerate(pdf_files[:10], 1):
        size_mb = pdf_file.stat().st_size / 1024 / 1024
        print(f"   {i:2d}. {pdf_file.name} ({size_mb:.1f} MB)")
    
    if total_files > 10:
        print(f"   ... 还有 {total_files - 10} 个文件")
    
    # 预估处理时间
    estimated_time = total_files * 5  # 每个文件约5分钟
    print(f"\n⏱️  预估处理时间: 约 {estimated_time} 分钟 ({estimated_time/60:.1f} 小时)")
    
    # 检查输出目录
    if os.path.exists(output_folder):
        existing_files = len(list(Path(output_folder).glob("*.md")))
        if existing_files > 0:
            print(f"\n📁 输出目录已存在，包含 {existing_files} 个Markdown文件")
    
    # 确认是否继续
    print("\n" + "=" * 50)
    confirm = input(f"确认要处理这 {total_files} 个PDF文件吗? (y/n): ").strip().lower()
    if confirm != 'y':
        logger.info("❌ 处理已取消")
        return
    
    # 询问是否恢复之前的进度
    resume = input("是否恢复之前的处理进度? (y/n): ").strip().lower() == 'y'
    
    print("\n🚀 开始批量处理...")
    print("💡 提示: 按 Ctrl+C 可以安全中断处理（进度会被保存）")
    print("=" * 50)
    
    try:
        processed_count = processor.process_batch(resume=resume)
        
        print("\n" + "=" * 50)
        print("🎉 批量处理完成!")
        print(f"✅ 成功处理: {processed_count} 个文件")
        print(f"📁 输出目录: {os.path.abspath(output_folder)}")
        
        # 显示最终统计
        if os.path.exists(output_folder):
            final_files = len(list(Path(output_folder).glob("*.md")))
            total_size = sum(f.stat().st_size for f in Path(output_folder).glob("*.md"))
            print(f"📊 输出统计: {final_files} 个Markdown文件，总大小 {total_size/1024/1024:.1f} MB")
        
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        logger.info("⏸️  处理被用户中断")
        logger.info("💾 进度已保存，下次可以选择恢复进度继续处理")
        print("=" * 50)
    except Exception as e:
        print("\n" + "=" * 50)
        logger.error(f"❌ 批量处理失败: {e}")
        logger.exception("详细错误信息:")
        print("=" * 50)


if __name__ == "__main__":
    main()
