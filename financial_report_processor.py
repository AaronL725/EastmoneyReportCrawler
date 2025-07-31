import asyncio
import os
import argparse
import logging
import logging.config
import inspect
import json
import shutil
import re
import base64
import requests
import time
import importlib
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from dotenv import load_dotenv

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

# MinerU 2.0 配置 - 不再需要配置文件，改用环境变量和参数配置
# 设置 MinerU 模型源（可选：huggingface, modelscope, local）
MINERU_MODEL_SOURCE = os.getenv("MINERU_MODEL_SOURCE", "local")
MINERU_DEVICE = os.getenv("MINERU_DEVICE", "cpu")  # cpu, cuda, cuda:0, npu, mps

# 设置环境变量
os.environ["MINERU_MODEL_SOURCE"] = MINERU_MODEL_SOURCE

# 检查 MinerU 2.0 安装状态
def check_mineru_installation():
    """检查 MinerU 2.0 是否正确安装"""
    try:
        # 尝试导入 MinerU 核心模块
        import mineru
        print(f"✅ MinerU 版本: {getattr(mineru, '__version__', '未知')}")
        return True
    except ImportError:
        print("❌ MinerU 未安装")
        print("安装命令: pip install -U 'mineru[core]'")
        return False
    except Exception as e:
        print(f"❌ MinerU 检查失败: {e}")
        return False

# 执行安装检查
HAS_MINERU = check_mineru_installation()

# 核心导入 - 基于四个示例文件的关键模块
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from lightrag.kg.shared_storage import initialize_pipeline_status

# 导入本地提示词文件
import prompt as local_prompt

# 尝试导入RAGAnything相关模块
try:
    from raganything import RAGAnything, RAGAnythingConfig
    from raganything.modalprocessors import (
        ImageModalProcessor,
        TableModalProcessor, 
        EquationModalProcessor,
    )
    HAS_RAGANYTHING = True
    print("✅ RAGAnything 模块导入成功")
except ImportError as e:
    HAS_RAGANYTHING = False
    print(f"❌ RAGAnything未安装或导入失败: {e}")
    print("提示: 请运行 'pip install raganything' 安装")

# 加载环境变量 - 只加载一次
load_dotenv(dotenv_path=".env", override=False)

# 全局配置
WORKING_DIR = "./financial_rag"
OUTPUT_DIR = "./financial_output"

# Ollama模型配置 - 从环境变量读取，支持动态配置
VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:3b-q4_K_M")        # 视觉理解
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")        # 文本嵌入
RERANK_MODEL = os.getenv("RERANK_MODEL_LOCAL", "qllama/bge-reranker-large:latest")  # 重排序
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "qwen3:1.7b-q4_K_M")  # 提取模型
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "qwen3:8b-q4_K_M")            # 最终问答模型

# 打印模型配置信息
print(f"\n研报处理模型配置:")
print(f"  - 视觉模型: {VISION_MODEL}")
print(f"  - 嵌入模型: {EMBEDDING_MODEL}")
print(f"  - 重排序模型: {RERANK_MODEL}")
print(f"  - 提取模型: {EXTRACTION_MODEL}")
print(f"  - 问答模型: {ANSWER_MODEL}")
if os.getenv("FORCE_MODEL"):
    print(f"  - 强制模型: {os.getenv('FORCE_MODEL')}")
print()


# Ollama服务配置
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
TIMEOUT = int(os.getenv("TIMEOUT", "300"))


def configure_logging():
    """配置日志系统 - 基于raganything_example.py的日志配置"""
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "financial_report_processor.log"))
    
    print(f"\n金融报告处理器日志文件: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    # 从环境变量获取日志配置
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # 默认10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))   # 默认5个备份
    
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(levelname)s: %(message)s",
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "console": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "file": {
                "formatter": "detailed",
                "class": "logging.handlers.RotatingFileHandler",
                "filename": log_file_path,
                "maxBytes": log_max_bytes,
                "backupCount": log_backup_count,
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "lightrag": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            },
        },
    })
    
    # 设置日志级别
    logger.setLevel(logging.INFO)
    set_verbose_debug(True)  

async def ollama_vision_complete_direct(prompt: str, system_prompt: str = None, image_data: str = None, **kwargs) -> str:
    """直接调用Ollama API的视觉模型函数"""
    import requests
    import json
    
    try:
        # 构建完整的提示词
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        # 构建请求数据 - 直接使用Ollama API格式
        api_url = f"{OLLAMA_HOST}/api/generate"
        data = {
            "model": VISION_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {"num_ctx": 4096}
        }
        
        # 如果有图像数据，添加到请求中
        if image_data:
            data["images"] = [image_data]
        
        # 发送请求，增加重试机制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 增加超时时间，处理大图片
                response = requests.post(api_url, json=data, timeout=TIMEOUT*2)
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise e
                logger.warning(f"视觉模型API调用尝试 {attempt + 1} 失败，5秒后重试...")
                await asyncio.sleep(5)
        
        # 解析响应
        try:
            result = response.json()
            description = result.get("response", "").strip()
            
            if description:
                return description
            else:
                return "无法生成图片描述"
        except json.JSONDecodeError as je:
            logger.error(f"JSON解析错误: {je}")
            # 尝试从响应文本中提取内容
            try:
                response_text = response.text.strip()
                logger.debug(f"原始响应内容: {response_text[:500]}...")
                
                # 如果响应包含有效内容但不是JSON格式，直接使用
                if response_text and len(response_text) > 10 and not response_text.startswith("ERROR"):
                    # 尝试提取可能的JSON部分
                    if "{" in response_text and "}" in response_text:
                        # 查找最后一个完整的JSON对象
                        json_start = response_text.rfind("{")
                        json_end = response_text.rfind("}") + 1
                        if json_start >= 0 and json_end > json_start:
                            json_part = response_text[json_start:json_end]
                            
                            # 尝试修复常见的JSON格式问题
                            json_part = re.sub(r',\s*([}\]])', r'\1', json_part)  # 移除末尾多余的逗号
                            json_part = re.sub(r'([^"\\])"([^":])', r'\1\\"\2', json_part)  # 转义引号
                            
                            try:
                                fixed_result = json.loads(json_part)
                                description = fixed_result.get("response", "").strip()
                                if description:
                                    logger.info("成功修复并解析JSON响应")
                                    return description
                            except json.JSONDecodeError:
                                logger.warning("JSON修复尝试失败")
                    
                    # 如果不是JSON，可能是纯文本响应，直接返回
                    if not response_text.startswith("{") and not response_text.startswith("["):
                        logger.info("使用纯文本响应作为图片描述")
                        return response_text
                    
                return "无法生成图片描述"
                    
            except Exception as text_error:
                logger.error(f"处理响应文本失败: {text_error}")
                return "图片分析失败: 响应处理错误"
                
    except Exception as e:
        logger.error(f"直接视觉模型API调用失败: {e}")
        return f"图片分析失败: {str(e)}"


async def ollama_vision_complete(prompt: str, system_prompt: str = None, history_messages: list = None, image_data: str = None, **kwargs) -> str:
    """视觉模型调用函数 - 包装直接API调用以保持接口兼容性"""
    return await ollama_vision_complete_direct(
        prompt=prompt,
        system_prompt=system_prompt,
        image_data=image_data,
        **kwargs
    )


class OllamaImageDescriber:
    """使用Ollama API描述图片内容的类"""
    
    def __init__(self, base_url: str = OLLAMA_HOST, model: str = VISION_MODEL):
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
    
    async def describe_image(self, image_path: str, prompt: str = None) -> str:
        """使用Ollama模型描述图片内容""" 
        
        # 使用本地提示词文件中的图表分析提示词
        if prompt is None:
            prompt = local_prompt.PROMPTS["financial_chart_analysis"]
        
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
                "stream": False,
                "options": {"num_ctx": 4096}
            }
            
            # 发送请求，增加重试机制
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    import requests
                    # 增加超时时间以适应大图片
                    response = requests.post(self.api_url, json=data, timeout=TIMEOUT*2)
                    response.raise_for_status()
                    break
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        raise e
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in 5 seconds...")
                    await asyncio.sleep(5)
            
            # 解析响应 - 增强错误处理
            try:
                import json
                result = response.json()
                description = result.get("response", "").strip()
                
                if description:
                    logger.info(f"Successfully described image: {os.path.basename(image_path)}")
                    return description
                else:
                    return f"[无法生成图片描述: {os.path.basename(image_path)}]"
            except json.JSONDecodeError as je:
                logger.error(f"图片描述JSON解析错误: {je}")
                # 尝试从响应文本中提取内容
                try:
                    response_text = response.text.strip()
                    logger.debug(f"图片描述原始响应: {response_text[:300]}...")
                    
                    # 如果响应包含有效内容但不是JSON格式，直接使用
                    if response_text and len(response_text) > 10 and not response_text.startswith("ERROR"):
                        # 尝试提取可能的JSON部分
                        if "{" in response_text and "}" in response_text:
                            # 查找最后一个完整的JSON对象
                            json_start = response_text.rfind("{")
                            json_end = response_text.rfind("}") + 1
                            if json_start >= 0 and json_end > json_start:
                                json_part = response_text[json_start:json_end]
                                
                                # 尝试修复常见的JSON格式问题
                                json_part = re.sub(r',\s*([}\]])', r'\1', json_part)  # 移除末尾多余的逗号
                                json_part = re.sub(r'([^"\\])"([^":])', r'\1\\"\2', json_part)  # 转义引号
                                
                                try:
                                    fixed_result = json.loads(json_part)
                                    description = fixed_result.get("response", "").strip()
                                    if description:
                                        logger.info(f"修复JSON后成功描述图片: {os.path.basename(image_path)}")
                                        return description
                                except json.JSONDecodeError:
                                    logger.warning("图片描述JSON修复尝试失败")
                        
                        # 如果不是JSON，可能是纯文本响应，直接返回
                        if not response_text.startswith("{") and not response_text.startswith("["):
                            logger.info(f"使用纯文本作为图片描述: {os.path.basename(image_path)}")
                            return response_text
                        
                    return f"[无法生成图片描述: {os.path.basename(image_path)}]"
                        
                except Exception as text_error:
                    logger.error(f"处理图片描述响应文本失败: {text_error}")
                    return f"[图片描述失败(响应处理错误): {os.path.basename(image_path)}]"
                
        except Exception as e:
            logger.error(f"Failed to describe image {image_path}: {e}")
            return f"[图片描述失败: {os.path.basename(image_path)}]"
    
    def describe_image_sync(self, image_path: str, prompt: str = None) -> str:
        """使用Ollama模型描述图片内容"""
        
        # 使用金融图表分析专用提示词
        if prompt is None:
            prompt = """
你是一位专业的金融图表分析师。请仔细观察这张图片，用中文详细描述其内容。请严格按照客观事实描述，不要进行主观解读。

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

请用具体的数字和描述词填充以上内容，不要使用占位符或方括号。
            """
        
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
                "stream": False,
                "options": {"num_ctx": 4096}
            }
            
            # 发送请求，增加重试机制
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    import requests
                    response = requests.post(self.api_url, json=data, timeout=120)
                    response.raise_for_status()
                    break
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        raise e
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in 5 seconds...")
                    time.sleep(5)
            
            # 解析响应
            try:
                result = response.json()
                description = result.get("response", "").strip()
                
                if description:
                    logger.info(f"Successfully described image: {os.path.basename(image_path)}")
                    return description
                else:
                    return f"[无法生成图片描述: {os.path.basename(image_path)}]"
            except json.JSONDecodeError as je:
                logger.error(f"图片描述JSON解析错误: {je}")
                # 尝试直接使用响应文本
                try:
                    response_text = response.text.strip()
                    if response_text and len(response_text) > 10 and not response_text.startswith("ERROR"):
                        # 如果不是JSON，可能是纯文本响应，直接返回
                        if not response_text.startswith("{") and not response_text.startswith("["):
                            logger.info(f"使用纯文本作为图片描述: {os.path.basename(image_path)}")
                            return response_text
                    return f"[无法生成图片描述: {os.path.basename(image_path)}]"
                except Exception as text_error:
                    logger.error(f"处理图片描述响应文本失败: {text_error}")
                    return f"[图片描述失败(响应处理错误): {os.path.basename(image_path)}]"
                
        except Exception as e:
            logger.error(f"Failed to describe image {image_path}: {e}")
            return f"[图片描述失败: {os.path.basename(image_path)}]"


def get_ollama_vision_model_func():
    """获取兼容 modalprocessors 的视觉模型函数"""
    async def vision_model_wrapper(prompt, system_prompt=None, history_messages=None, image_data=None, **kwargs):
        # 确保不会有重复参数
        vision_kwargs = {
            "host": OLLAMA_HOST,
            "options": {"num_ctx": 4096},
            "timeout": TIMEOUT,
        }
        
        # 只添加不冲突的kwargs，并排除可能导致冲突的参数
        for key, value in kwargs.items():
            if key not in vision_kwargs and key not in ["model"]:
                vision_kwargs[key] = value
        
        # 确保不包含model参数，避免与内部逻辑冲突
        vision_kwargs.pop("model", None)
        
        return await ollama_vision_complete(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            image_data=image_data,
            **vision_kwargs
        )
    
    return vision_model_wrapper


async def ollama_embedding_func(texts: List[str]) -> List[List[float]]:
    """嵌入模型调用函数"""
    try:
        return await ollama_embed(
            texts=texts,
            embed_model=EMBEDDING_MODEL,
            host=OLLAMA_HOST,
        )
    except Exception as e:
        logger.error(f"嵌入模型调用失败: {e}")
        # 返回默认维度的零向量
        return [[0.0] * 1024 for _ in texts]


async def simple_ollama_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    """根据任务类型动态选择Ollama模型的调用函数"""
    # 支持的模型配置
    extraction_model = os.getenv("EXTRACTION_MODEL", "qwen3:1.7b-q4_K_M")
    answer_model = os.getenv("ANSWER_MODEL", "qwen3:8b-q4_K_M")
    # 允许外部直接指定模型
    model_name = kwargs.pop("model", None)
    # 任务类型优先级：task/llm_task > model > 默认
    task = kwargs.pop("task", None) or kwargs.pop("llm_task", None)
    if not model_name:
        if task in ("extract", "extraction", "entity_extraction", "relation_extraction"):
            model_name = extraction_model
        elif task in ("answer", "qa", "question_answering", "generate", "completion"):
            model_name = answer_model
        else:
            # 默认用问答模型
            model_name = answer_model
    # 彻底清理参数以避免冲突
    safe_kwargs = {}
    safe_internal_params = ["hashing_kv", "embedding_func", "semaphore"]
    for key in safe_internal_params:
        if key in kwargs:
            safe_kwargs[key] = kwargs[key]
    safe_kwargs.update({
        "host": OLLAMA_HOST,
        "options": {"num_ctx": 8192},
        "timeout": TIMEOUT,
    })
    safe_kwargs.pop("model", None)
    try:
        api_url = f"{OLLAMA_HOST}/api/generate"
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        data = {
            "model": model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {"num_ctx": 8192}
        }
        response = requests.post(api_url, json=data, timeout=TIMEOUT)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        logger.error(f"直接API调用失败，回退到LightRAG调用: {e}")
        return await ollama_model_complete(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            **safe_kwargs
        )


async def modal_ollama_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    """为模态处理器定制的ollama模型函数 - 使用环境变量配置的模型"""
    
    # 从环境变量获取模型配置，默认使用回答模型
    model_name = os.getenv("FORCE_MODEL", ANSWER_MODEL)
    logger.debug(f"模态处理器使用模型: {model_name}")
    
    # 使用直接API调用避免LightRAG参数冲突
    try:
        api_url = f"{OLLAMA_HOST}/api/generate"
        
        # 构建完整的提示词
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        data = {
            "model": model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {"num_ctx": 4096}  # 模态处理使用稍小的上下文
        }
        
        response = requests.post(api_url, json=data, timeout=TIMEOUT)
        response.raise_for_status()
        result = response.json()
        
        return result.get("response", "").strip()
        
    except Exception as e:
        logger.error(f"模态处理器直接API调用失败，回退到LightRAG调用: {e}")
        
        # 彻底清理参数以避免冲突
        safe_kwargs = {}
        
        # 只保留安全的内部参数
        safe_internal_params = ["hashing_kv", "embedding_func", "semaphore"]
        for key in safe_internal_params:
            if key in kwargs:
                safe_kwargs[key] = kwargs[key]
        
        # 添加自定义参数，但排除可能导致冲突的参数
        safe_kwargs.update({
            "host": OLLAMA_HOST,
            "options": {"num_ctx": 4096},  # 模态处理使用稍小的上下文
            "timeout": TIMEOUT,
        })
        
        # 确保不包含model参数，避免与LightRAG内部逻辑冲突
        safe_kwargs.pop("model", None)
        
        # 通过修改hashing_kv中的配置来动态设置模型名
        if "hashing_kv" in safe_kwargs and hasattr(safe_kwargs["hashing_kv"], "global_config"):
            # 临时保存原始模型名
            original_model = safe_kwargs["hashing_kv"].global_config.get("llm_model_name", ANSWER_MODEL)
            # 设置新的模型名
            safe_kwargs["hashing_kv"].global_config["llm_model_name"] = model_name
            logger.debug(f"模态处理器通过hashing_kv设置模型: {original_model} -> {model_name}")
        
        return await ollama_model_complete(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            **safe_kwargs
        )


async def ollama_rerank_func(query: str, documents: List[Dict], top_n: int = 10, **kwargs) -> List[Dict]:
    """本地重排序函数 - 使用配置的重排序模型"""
    try:
        # 构建重排序的输入格式
        doc_texts = [doc.get("content", str(doc)) for doc in documents]
        
        # 使用配置的重排序模型进行评分
        scores = []
        for doc_text in doc_texts:
            # 使用本地提示词文件中的重排序评分提示词
            score_prompt = local_prompt.PROMPTS["financial_rerank_score"].format(
                query=query,
                doc_text=doc_text
            )
            
            try:
                # 使用直接API调用
                api_url = f"{OLLAMA_HOST}/api/generate"
                data = {
                    "model": RERANK_MODEL,  # 使用配置的重排序模型
                    "prompt": score_prompt,
                    "system": "你是一位专业的金融文档相关性评估专家，擅长判断文档与用户金融查询的相关程度。请客观公正地评估，只给出0-1之间的数值评分，不要添加任何其他文字。",
                    "stream": False,
                    "options": {"num_ctx": 2048}
                }
                
                import requests
                response = requests.post(api_url, json=data, timeout=60)
                response.raise_for_status()
                result = response.json()
                score_response = result.get("response", "").strip()
                
                # 尝试提取数字分数
                import re
                score_match = re.search(r'(\d+\.?\d*)', score_response)
                score = float(score_match.group(1)) if score_match else 0.5
                score = max(0.0, min(1.0, score))  # 限制在0-1范围内
            except:
                score = 0.5  # 默认分数
                
            scores.append(score)
        
        # 根据分数排序并返回top_n
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        result = []
        for doc, score in scored_docs[:top_n]:
            result_doc = doc.copy()
            result_doc["rerank_score"] = score
            result.append(result_doc)
            
        return result
        
    except Exception as e:
        logger.error(f"重排序失败: {e}")
        return documents[:top_n]


class FinancialReportProcessor:
    """金融报告处理器主类"""
    
    def __init__(self, working_dir: str = WORKING_DIR):
        self.working_dir = working_dir
        
        # 先应用补丁函数，避免后续初始化出错
        self._patch_raganything_mineru_check()
        
        # 确保所有需要的目录都存在
        os.makedirs(working_dir, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.working_dir, "financial_reports"), exist_ok=True)
        
        # 确保数据目录存在
        reports_dir = os.path.join(self.working_dir, "financial_reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # 创建必要的空文件，确保文件存在
        for file_name in ["kv_store_doc_status.json", "kv_store_full_docs.json", "kv_store_text_chunks.json"]:
            file_path = os.path.join(reports_dir, file_name)
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    f.write("{}")
        
        self.rag = None
        self.modal_processors = {}
        
        # 初始化图片描述器
        self.image_describer = OllamaImageDescriber(OLLAMA_HOST, VISION_MODEL)
    
    def _patch_raganything_mineru_check(self):
        """修复 RAGAnything 对 MinerU 的检测，强制设置为已安装状态"""
        if HAS_RAGANYTHING:
            try:
                # 导入需要修改的模块
                import raganything.utils
                import raganything
                
                # 动态修改 RAGAnything 的检测函数，使其始终返回 True
                original_check = raganything.utils.check_mineru_installation
                raganything.utils.check_mineru_installation = lambda: True
                
                # 如果有__mineru_installed__属性，也设置为True
                if hasattr(raganything, "__mineru_installed__"):
                    raganything.__mineru_installed__ = True
                    
                # 补丁其他可能使用该检查的地方
                try:
                    # 尝试导入处理文档的类
                    import importlib
                    document_processor = importlib.import_module("raganything.docprocessor")
                    if hasattr(document_processor, "DocumentProcessor"):
                        processor_class = getattr(document_processor, "DocumentProcessor")
                        if hasattr(processor_class, "_check_mineru"):
                            processor_class._check_mineru = lambda self: True
                except (ImportError, AttributeError):
                    pass
                
                # 确保配置类也知道MinerU已安装
                try:
                    from raganything import RAGAnythingConfig
                    original_init = RAGAnythingConfig.__init__
                    
                    def patched_init(self, *args, **kwargs):
                        result = original_init(self, *args, **kwargs)
                        self.mineru_installed = True
                        return result
                    
                    RAGAnythingConfig.__init__ = patched_init
                except Exception:
                    pass
                
                logger.info("✅ 已修复 RAGAnything 的 MinerU 检测功能")
            except Exception as e:
                logger.warning(f"无法修复 RAGAnything 的 MinerU 检测: {e}")
    
    async def initialize(self):
        """初始化处理器"""
        
        # 清理旧数据文件
        await self._clean_old_data()
        
        # 初始化RAG系统
        await self._initialize_rag()
        
        # 不再需要模态处理器，图片处理直接通过 Ollama 完成
        logger.info("✅ 处理器初始化完成 - 使用直接Ollama图片处理模式")
    
    async def _clean_old_data(self):
        """清理旧的数据文件"""
        # 使用正确的工作目录路径
        reports_dir = os.path.join(self.working_dir, "financial_reports")
        
        files_to_delete = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_doc_status.json", 
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json",
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json",
        ]
        
        for file in files_to_delete:
            file_path = os.path.join(reports_dir, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"删除旧文件: {file_path}")
    
    async def save_content_to_file(self, content: Union[str, dict, list], filename: str, directory: str = OUTPUT_DIR) -> str:
        """保存内容到文件，确保目录存在"""
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, filename)
        
        try:
            if isinstance(content, (dict, list)):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(content, f, ensure_ascii=False, indent=2)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(str(content))
            logger.info(f"内容已保存到: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"保存内容到文件失败: {e}")
            return ""
    
    async def _initialize_rag(self, cleanup_old_data: bool = True):
        """初始化RAG系统 - 支持是否清理旧数据的选项"""
        reports_dir = os.path.join(self.working_dir, "knowledge_base")
        os.makedirs(reports_dir, exist_ok=True)
        
        logger.info(f"使用知识库目录: {reports_dir}")
        
        # 如果需要清理，则删除旧数据
        if cleanup_old_data:
            await self._clean_old_data()
        
        # 确保必要的KV存储文件存在
        for filename in ["kv_store_doc_status.json", "kv_store_full_docs.json", "kv_store_text_chunks.json"]:
            file_path = os.path.join(reports_dir, filename)
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    f.write("{}")
                logger.info(f"创建空KV存储文件: {file_path}")
        
        # 测试嵌入功能并获取维度
        test_embedding = await ollama_embedding_func(["测试文本"])
        embedding_dim = len(test_embedding[0])
        logger.info(f"检测到嵌入维度: {embedding_dim}")
        
        # 获取本地提示词并应用
        get_local_prompts()
        
        # 创建LightRAG实例，使用简化的目录结构
        logger.info(f"LightRAG将使用工作目录: {reports_dir}")
        # 包装llm_model_func，自动识别任务类型
        async def llm_model_func_with_task(prompt, system_prompt=None, history_messages=None, **kwargs):
            # 优先级1：如果明确指定了task或model，直接用
            if "task" in kwargs and kwargs["task"]:
                return await simple_ollama_model_func(prompt, system_prompt, history_messages, **kwargs)
            if "model" in kwargs and kwargs["model"]:
                return await simple_ollama_model_func(prompt, system_prompt, history_messages, **kwargs)
            # 优先级2：如果system_prompt或prompt中包含模型名关键词，强制切换
            check_text = (system_prompt or "") + "\n" + (prompt or "")
            extraction_model = os.getenv("EXTRACTION_MODEL", "qwen3:1.7b-q4_K_M")
            answer_model = os.getenv("ANSWER_MODEL", "qwen3:8b-q4_K_M")
            extraction_keywords = ["实体提取", "关系提取", "entity extraction", "relation extraction", "信息抽取", "extraction task", "extract entities", "extract relations"]
            answer_keywords = ["问答", "回答", "answer", "qa", "question", "generate", "completion"]
            # 如果文本中出现了提取模型名或明显的实体提取关键词，强制用提取模型
            if extraction_model in check_text or any(word in check_text.lower() for word in extraction_keywords):
                kwargs["model"] = extraction_model
                kwargs["task"] = "extract"
                return await simple_ollama_model_func(prompt, system_prompt, history_messages, **kwargs)
            # 只有明确是问答/生成才用answer模型
            if answer_model in check_text or any(word in check_text.lower() for word in answer_keywords):
                kwargs["model"] = answer_model
                kwargs["task"] = "answer"
                return await simple_ollama_model_func(prompt, system_prompt, history_messages, **kwargs)
            # 默认用提取模型，保证抽取阶段不会误用answer模型
            kwargs["model"] = extraction_model
            kwargs["task"] = "extract"
            return await simple_ollama_model_func(prompt, system_prompt, history_messages, **kwargs)

        self.rag = LightRAG(
            working_dir=reports_dir,
            workspace="",  # 避免额外嵌套
            llm_model_func=llm_model_func_with_task,
            llm_model_name=ANSWER_MODEL,  # 默认使用高质量回答模型
            llm_model_max_token_size=8192,
            llm_model_kwargs={
                "host": OLLAMA_HOST,
                "options": {"num_ctx": 8192},
                "timeout": TIMEOUT,
            },
            rerank_model_func=ollama_rerank_func,  # 使用配置的重排序函数
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dim,
                max_token_size=8192,
                func=lambda texts: ollama_embed(
                    texts,
                    embed_model=EMBEDDING_MODEL,
                    host=OLLAMA_HOST,
                ),
            ),
        )
        
        await self.rag.initialize_storages()
        await initialize_pipeline_status()
        
        if cleanup_old_data:
            logger.info("RAG系统初始化完成")
        else:
            logger.info("RAG系统初始化完成（保留已有数据）")
    
    async def check_existing_data(self) -> bool:
        """检查是否已有处理好的数据，可以直接开始查询"""
        try:
            # 使用简化的目录结构
            reports_dir = os.path.join(self.working_dir, "knowledge_base")
            
            # 检查关键的知识图谱文件
            required_files = [
                "kv_store_full_docs.json",
                "kv_store_text_chunks.json", 
                "vdb_entities.json",
                "vdb_relationships.json"
            ]
            
            existing_files = []
            for file_name in required_files:
                file_path = os.path.join(reports_dir, file_name)
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path)
                    if size > 10:  # 文件不为空（大于10字节）
                        existing_files.append((file_name, size))
                        logger.info(f"✅ 发现已存在数据: {file_name} (大小: {size} bytes)")
            
            # 检查输出目录中的处理结果
            output_files = []
            if os.path.exists(OUTPUT_DIR):
                for file in os.listdir(OUTPUT_DIR):
                    if file.endswith(('.md', '.json')) and os.path.getsize(os.path.join(OUTPUT_DIR, file)) > 0:
                        output_files.append(file)
                        logger.info(f"✅ 发现输出文件: {file}")
            
            # 如果有足够的数据文件存在，说明之前已经处理过文档
            if len(existing_files) >= 2:  # 至少有2个关键文件存在
                logger.info(f"📚 发现已存在的知识库数据 ({len(existing_files)}/{len(required_files)} 个文件)")
                if output_files:
                    logger.info(f"📄 发现 {len(output_files)} 个处理结果文件")
                return True
            else:
                logger.info("❌ 未发现完整的知识库数据")
                return False
                
        except Exception as e:
            logger.error(f"检查已存在数据失败: {e}")
            return False
    
    async def load_existing_data(self) -> bool:
        """加载已存在的数据到RAG系统"""
        try:
            # 初始化RAG系统（不清理数据）
            await self._initialize_rag(cleanup_old_data=False)
            
            # 检查知识图谱状态
            await self._check_knowledge_graph_status()
            
            logger.info("✅ 已成功加载现有数据到RAG系统")
            return True
            
        except Exception as e:
            logger.error(f"加载已存在数据失败: {e}")
            return False
    

    async def _check_knowledge_graph_status(self):
        """检查知识图谱构建状态"""
        try:
            # 使用简化的目录结构
            reports_dir = os.path.join(self.working_dir, "knowledge_base")
            
            # 检查关键文件是否存在
            kg_files = {
                "graph_chunk_entity_relation.graphml": "知识图谱文件",
                "vdb_entities.json": "实体向量数据库",
                "vdb_relationships.json": "关系向量数据库",
                "kv_store_full_docs.json": "文档存储",
                "kv_store_text_chunks.json": "文本块存储"
            }
            
            for file_name, description in kg_files.items():
                file_path = os.path.join(reports_dir, file_name)
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path)
                    logger.info(f"✅ {description}: {file_path} (大小: {size} bytes)")
                else:
                    logger.warning(f"❌ {description}: {file_path} 不存在")
                    
        except Exception as e:
            logger.error(f"检查知识图谱状态失败: {e}")

    
    def cleanup_all_output_files(self):
        """
        清理 financial_output 目录下所有无用 .md 文件和空文件夹（递归处理）
        - 只保留 *_enhanced.md 文件，其余 .md 一律删除
        - 递归删除所有空文件夹
        - 处理隐藏文件和 .DS_Store
        """
        import shutil
        if not os.path.exists(OUTPUT_DIR):
            logger.info("输出目录不存在，无需清理")
            return
        logger.info("🧹 开始递归清理 financial_output 目录...")
        cleaned_files = []
        cleaned_dirs = []

        # 递归删除所有非 _enhanced.md 的 .md 文件
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for fname in files:
                fpath = os.path.join(root, fname)
                # 删除所有 .md 且不是 _enhanced.md 的文件
                if fname.endswith('.md') and not fname.endswith('_enhanced.md'):
                    try:
                        os.remove(fpath)
                        cleaned_files.append(fpath)
                        logger.info(f"🧹 删除老的 markdown 文件: {fpath}")
                    except Exception as e:
                        logger.warning(f"删除 {fpath} 失败: {e}")
                # 删除 macOS 下的 .DS_Store
                elif fname == '.DS_Store':
                    try:
                        os.remove(fpath)
                        cleaned_files.append(fpath)
                        logger.info(f"🧹 删除无用文件: {fpath}")
                    except Exception as e:
                        logger.warning(f"删除 {fpath} 失败: {e}")

        # 递归删除所有空文件夹（自底向上）
        for root, dirs, files in os.walk(OUTPUT_DIR, topdown=False):
            for d in dirs:
                dpath = os.path.join(root, d)
                # 跳过隐藏文件夹
                if d.startswith('.'):
                    continue
                try:
                    # 只要目录为空就删
                    if not os.listdir(dpath):
                        os.rmdir(dpath)
                        cleaned_dirs.append(dpath)
                        logger.info(f"🧹 删除空文件夹: {dpath}")
                except Exception as e:
                    logger.warning(f"删除空文件夹 {dpath} 失败: {e}")

        if cleaned_files or cleaned_dirs:
            logger.info(f"✅ 清理完成: 删除了 {len(cleaned_files)} 个文件和 {len(cleaned_dirs)} 个空文件夹")
        else:
            logger.info("✅ 无需清理，目录已干净")
    
    
    async def process_markdown_with_image_descriptions(self, md_content: str, image_dir: str) -> str:
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
                try:
                    # 直接使用Ollama模型描述图片，不通过LightRAG
                    description = self.image_describer.describe_image_sync(full_image_path)
                    # 格式化描述文本
                    return f"\n**[图片描述]**: {description}\n"
                except Exception as e:
                    logger.error(f"处理图片描述失败 {image_filename}: {e}")
                    return f"\n**[图片描述失败]**: {image_filename}\n"
            else:
                logger.warning(f"Image file not found: {full_image_path}")
                return f"\n**[图片未找到]**: {image_path}\n"
        
        # 匹配markdown中的图片语法 ![alt](path)
        image_pattern = r'!\[.*?\]\((.*?)\)'
        processed_content = re.sub(image_pattern, replace_image_with_description, md_content)
        
        return processed_content
    
    async def _process_images_with_vision_model(self, md_content: str, image_dir: str, file_name: str) -> str:
        """使用视觉模型处理图片并生成描述"""
        if not os.path.exists(image_dir):
            logger.warning(f"图片目录不存在: {image_dir}")
            return md_content
            
        logger.info(f"开始处理图片目录: {image_dir}")
        
        # 使用pdf_batch_processor.py的处理方法
        try:
            enhanced_content = await self.process_markdown_with_image_descriptions(md_content, image_dir)
            logger.info(f"已完成图片描述处理")
            return enhanced_content
        except Exception as e:
            logger.error(f"图片描述处理失败: {e}")
            return md_content
    
    async def _describe_image_with_vision_model(self, image_path: str) -> str:
        """使用视觉模型描述单个图片"""
        try:
            # 使用独立的图片描述器，避免LightRAG的hashing_kv问题
            description = await self.image_describer.describe_image(image_path)
            
            if description and not description.startswith("[图片") and not description.startswith("图片分析失败"):
                return description.strip()
            else:
                return f"[无法生成图片描述: {os.path.basename(image_path)}]"
                
        except Exception as e:
            logger.error(f"视觉模型描述图片失败 {image_path}: {e}")
            return f"[图片描述失败: {os.path.basename(image_path)}]"
    
    async def process_document(self, file_path: str) -> bool:
        """处理文档"""
        try:
            logger.info(f"开始处理文档: {file_path}")
            # 检查文件是否存在
            if not os.path.exists(file_path):
                logger.error(f"文件不存在: {file_path}")
                return False
            # PDF+MinerU流程
            if file_path.lower().endswith('.pdf') and HAS_MINERU:
                try:
                    logger.info("使用 MinerU 直接处理 PDF 文件...")
                    from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
                    from mineru.data.data_reader_writer import FileBasedDataWriter
                    from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
                    from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
                    from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
                    from mineru.utils.enum_class import MakeMode
                    
                    # 读取PDF文件
                    file_name = str(Path(file_path).stem)
                    pdf_bytes = read_fn(file_path)
                    
                    # 转换PDF字节
                    new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, 0, None)
                    
                    # 使用pipeline模式分析
                    infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
                        [new_pdf_bytes], ['ch'], parse_method="auto", formula_enable=True, table_enable=True
                    )
                    
                    # 准备输出环境
                    local_image_dir, local_md_dir = prepare_env(OUTPUT_DIR, file_name, "auto")
                    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
                    
                    # 处理结果
                    model_list = infer_results[0]
                    images_list = all_image_lists[0]
                    pdf_doc = all_pdf_docs[0]
                    _lang = lang_list[0]
                    _ocr_enable = ocr_enabled_list[0]
                    
                    # 转换为中间JSON格式
                    middle_json = pipeline_result_to_middle_json(
                        model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable, True
                    )
                    
                    pdf_info = middle_json["pdf_info"]
                    
                    # 生成markdown内容
                    image_dir = str(os.path.basename(local_image_dir))
                    md_content_str = pipeline_union_make(pdf_info, MakeMode.MM_MD, image_dir)
                    
                    # 保存生成的Markdown到文件
                    md_output_path = os.path.join(OUTPUT_DIR, f"{file_name}.md")
                    with open(md_output_path, 'w', encoding='utf-8') as f:
                        f.write(md_content_str)
                    logger.info(f"Markdown内容已保存到: {md_output_path}")
                    
                    # 处理图片：使用Ollama直接解析图片并替换markdown中的图片链接
                    logger.info("开始使用Ollama处理图片...")
                    enhanced_content = await self._process_images_with_vision_model(
                        md_content_str, local_image_dir, file_name
                    )
                    enhanced_md_path = os.path.join(OUTPUT_DIR, f"{file_name}_enhanced.md")
                    with open(enhanced_md_path, 'w', encoding='utf-8') as f:
                        f.write(enhanced_content)
                    logger.info(f"增强版Markdown已保存到: {enhanced_md_path}")
                    await self.rag.ainsert(enhanced_content)
                    logger.info("✅ 图片处理完成，内容已插入到LightRAG")
                    logger.info("⏳ 等待知识图谱构建完成...")
                    await asyncio.sleep(3)
                    await self._check_knowledge_graph_status()
                    
                    # 清理图像目录
                    try:
                        if os.path.exists(local_image_dir):
                            shutil.rmtree(local_image_dir)
                            logger.info(f"Processed images with Ollama and cleaned up {local_image_dir}")
                    except Exception as e:
                        logger.error(f"Failed to clean up image directory {local_image_dir}: {e}")
                    try:
                        if os.path.exists(local_md_dir) and local_md_dir != local_image_dir:
                            md_files = [f for f in os.listdir(local_md_dir) if not f.startswith('.') and os.path.isfile(os.path.join(local_md_dir, f))]
                            if md_files:
                                for md_file in md_files:
                                    try:
                                        os.remove(os.path.join(local_md_dir, md_file))
                                    except Exception:
                                        pass
                                logger.info(f"已清理Markdown临时文件: {local_md_dir}")
                    except Exception as e:
                        logger.warning(f"清理临时文件失败: {e}")
                    
                    logger.info("✅ 使用 MinerU 直接处理 PDF 完成")
                    return True
                    
                except Exception as mineru_error:
                    logger.error(f"MinerU 直接处理失败: {mineru_error}")
                    return False
            logger.warning("❌ 仅支持PDF+MinerU流程，其他类型文件暂不处理")
            return False
        except Exception as e:
            logger.error(f"文档处理失败: {e}")
            return False
    
    async def query_document(self, question: str) -> str:
        """查询文档 - 使用 hybrid 模式和流式输出"""
        try:
            logger.info(f"查询问题: {question}")
            
            # 使用 hybrid 模式和流式输出
            query_param = QueryParam(
                mode="hybrid",
                stream=True,
            )
            
            # 执行查询
            response = await self.rag.aquery(question, param=query_param)
            
            # 处理流式响应
            if inspect.isasyncgen(response):
                result_chunks = []
                async for chunk in response:
                    result_chunks.append(chunk)
                response = "".join(result_chunks)
            
            # 检查响应是否为空
            if not response or "抱歉，您没有提供文章内容" in response:
                return "抱歉，我无法在当前文档中找到相关信息来回答您的问题。请尝试询问与文档内容更相关的问题。"
            
            logger.info("查询完成")
            return response
            
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return f"查询出错: {str(e)}"
    
    async def query_with_financial_context(self, question: str, context_data: Optional[Dict] = None) -> str:
        """金融查询"""
        try:
            # 使用本地提示词文件中的金融系统提示
            financial_system_prompt = local_prompt.PROMPTS["financial_system"]
            
            if context_data and HAS_RAGANYTHING:
                # 如果有额外的财务数据，使用多模态查询
                multimodal_content = []
                
                # 添加表格数据
                if "table_data" in context_data:
                    multimodal_content.append({
                        "type": "table",
                        "table_data": context_data["table_data"],
                        "table_caption": context_data.get("table_caption", "财务数据表")
                    })
                
                # 添加公式
                if "formula" in context_data:
                    multimodal_content.append({
                        "type": "equation", 
                        "latex": context_data["formula"],
                        "equation_caption": context_data.get("formula_caption", "财务计算公式")
                    })
                
                if multimodal_content:
                    # 使用多模态查询
                    enhanced_question = f"{financial_system_prompt}\n\n{question}"
                    # 注意：这里需要RAGAnything实例，实际实现时需要调整
                    logger.info("使用多模态查询处理金融问题")
            
            # 标准查询流程
            response = await self.query_document(question)
            
            return response
            
        except Exception as e:
            logger.error(f"金融专业查询失败: {e}")
            return f"金融查询出错: {str(e)}"
    
    async def process_documents_batch(self, report_directory: str = None) -> Dict[str, bool]:
        """增量批量处理报告文件夹下的所有文档"""
        if report_directory is None:
            # 默认使用同级的Report文件夹
            current_dir = os.path.dirname(os.path.abspath(__file__))
            report_directory = os.path.join(current_dir, "Report")
        
        logger.info(f"开始批量处理报告文件夹: {report_directory}")
        
        # 检查Report文件夹是否存在
        if not os.path.exists(report_directory):
            logger.error(f"报告文件夹不存在: {report_directory}")
            return {}
        
        # 递归收集所有支持的文件
        supported_extensions = {'.pdf', '.txt', '.md'}
        all_files = []
        
        for root, dirs, files in os.walk(report_directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in supported_extensions:
                    all_files.append(file_path)
        
        logger.info(f"发现 {len(all_files)} 个可处理的文件")
        
        # 检查哪些文件已经处理过（增量处理逻辑）
        processed_files = set()
        new_files = []
        
        for file_path in all_files:
            file_name = Path(file_path).stem
            enhanced_md_path = os.path.join(OUTPUT_DIR, f"{file_name}_enhanced.md")
            simple_md_path = os.path.join(OUTPUT_DIR, f"{file_name}.md")
            
            # 如果存在增强版或简单版的markdown文件，认为已处理
            if os.path.exists(enhanced_md_path) or os.path.exists(simple_md_path):
                processed_files.add(file_path)
                logger.info(f"⏭️  文件已处理，跳过: {os.path.basename(file_path)}")
            else:
                new_files.append(file_path)
        
        logger.info(f"需要处理的新文件: {len(new_files)} 个")
        logger.info(f"已跳过的文件: {len(processed_files)} 个")
        
        if not new_files:
            logger.info("✅ 所有文件都已处理完成")
            return {file_path: True for file_path in all_files}
        
        # 按顺序处理新文件（非异步，避免资源冲突）
        results = {}
        processed_count = 0
        failed_count = 0
        
        # 先将已处理的文件标记为成功
        for file_path in processed_files:
            results[file_path] = True
        
        for i, file_path in enumerate(new_files, 1):
            logger.info(f"处理进度: {i}/{len(new_files)} - {os.path.basename(file_path)}")
            
            try:
                # 处理单个文档
                success = await self.process_document(file_path)
                results[file_path] = success
                
                if success:
                    processed_count += 1
                    logger.info(f"✅ 成功处理: {os.path.basename(file_path)}")
                else:
                    failed_count += 1
                    logger.error(f"❌ 处理失败: {os.path.basename(file_path)}")
                
                # 在文档之间添加短暂延迟，避免资源冲突
                if i < len(new_files):
                    logger.info("⏳ 等待3秒后处理下一个文件...")
                    await asyncio.sleep(3)
                    
            except Exception as e:
                logger.error(f"处理文档时发生异常 {os.path.basename(file_path)}: {e}")
                results[file_path] = False
                failed_count += 1
        
        # 最终统计
        total_processed = len(processed_files) + processed_count
        total_files = len(all_files)
        
        logger.info(f"批量处理完成!")
        logger.info(f"  总文件数: {total_files}")
        logger.info(f"  已处理(跳过): {len(processed_files)}")
        logger.info(f"  新处理成功: {processed_count}")
        logger.info(f"  处理失败: {failed_count}")
        logger.info(f"  整体成功率: {total_processed}/{total_files} ({total_processed/total_files*100:.1f}%)")
        
        return results
    
    async def process_single_file_to_knowledge_base(self, file_path: str) -> bool:
        """处理单个文件并添加到知识库（用于增量添加）"""
        try:
            # 检查文件是否已经处理过
            file_name = Path(file_path).stem
            enhanced_md_path = os.path.join(OUTPUT_DIR, f"{file_name}_enhanced.md")
            simple_md_path = os.path.join(OUTPUT_DIR, f"{file_name}.md")
            
            if os.path.exists(enhanced_md_path) or os.path.exists(simple_md_path):
                logger.info(f"文件已处理，直接加载到知识库: {os.path.basename(file_path)}")
                
                # 读取已处理的markdown内容
                content_path = enhanced_md_path if os.path.exists(enhanced_md_path) else simple_md_path
                with open(content_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 直接插入到知识库
                await self.rag.ainsert(content)
                logger.info(f"✅ 已将处理过的内容添加到知识库: {os.path.basename(file_path)}")
                return True
            else:
                # 处理新文件
                logger.info(f"处理新文件: {os.path.basename(file_path)}")
                return await self.process_document(file_path)
                
        except Exception as e:
            logger.error(f"单文件处理失败 {file_path}: {e}")
            return False
    
    async def update_knowledge_base_incrementally(self, report_directory: str = None) -> Dict[str, bool]:
        """增量更新知识库"""
        import re
        if report_directory is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            report_directory = os.path.join(current_dir, "Report")

        logger.info(f"增量更新知识库，扫描目录: {report_directory}")

        # 1. 获取 financial_output 目录下所有已处理文件的文件名（不含扩展名，含_enhanced）
        processed_names = set()
        if os.path.exists(OUTPUT_DIR):
            for fname in os.listdir(OUTPUT_DIR):
                if fname.endswith('.md'):
                    # 支持 _enhanced.md 和 .md
                    name = fname[:-3]  # 去掉.md
                    if name.endswith('_enhanced'):
                        name = name[:-9]  # 去掉_enhanced
                    processed_names.add(name)

        # 2. 遍历 Report 及其子目录，收集所有支持的文件
        supported_extensions = {'.pdf', '.txt', '.md'}
        all_files = []
        file_name_map = {}  # stem -> full path
        for root, dirs, files in os.walk(report_directory):
            for file in files:
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in supported_extensions:
                    file_path = os.path.join(root, file)
                    stem = Path(file).stem
                    all_files.append(file_path)
                    file_name_map[stem] = file_path

        if not all_files:
            logger.warning(f"在目录 {report_directory} 中未找到可处理的文件")
            return {}

        # 3. 只处理未在 processed_names 中的文件
        new_files = []
        for file_path in all_files:
            stem = Path(file_path).stem
            if stem not in processed_names:
                new_files.append(file_path)

        logger.info(f"发现 {len(new_files)} 个需要增量处理的新文件，{len(all_files)-len(new_files)} 个已处理文件将跳过。")

        results = {}
        new_processed = 0
        failed = 0
        for i, file_path in enumerate(new_files, 1):
            logger.info(f"增量处理进度: {i}/{len(new_files)} - {os.path.basename(file_path)}")
            try:
                success = await self.process_document(file_path)
                results[file_path] = success
                if success:
                    new_processed += 1
                    logger.info(f"✅ 新处理文件: {os.path.basename(file_path)}")
                else:
                    failed += 1
                    logger.error(f"❌ 处理失败: {os.path.basename(file_path)}")
                if i < len(new_files):
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"处理文件异常 {os.path.basename(file_path)}: {e}")
                results[file_path] = False
                failed += 1

        logger.info(f"增量更新完成! 新处理: {new_processed}，失败: {failed}，总新文件: {len(new_files)}")
        return results

    async def finalize(self):
        """清理资源"""
        if self.rag:
            try:
                # 确保所有 LightRAG 缓存操作已完成
                if hasattr(self.rag, 'llm_response_cache') and self.rag.llm_response_cache:
                    await self.rag.llm_response_cache.index_done_callback()
                
                # 结束所有存储
                await self.rag.finalize_storages()
                logger.info("RAG系统资源清理完成")
            except Exception as e:
                logger.error(f"RAG系统资源清理失败: {e}")
        
        # 等待一段时间以确保所有处理都已完成
        await asyncio.sleep(1)
        
        logger.info("程序结束 - 图片目录已在处理时清理完成")


def get_local_prompts():
    """获取本地提示词，用于替换LightRAG库中的默认提示词"""
    from lightrag.prompt import PROMPTS as lightrag_prompts
    
    # 用本地提示词替换默认提示词
    for key, value in local_prompt.PROMPTS.items():
        lightrag_prompts[key] = value
    
    logger.info(f"已加载本地提示词，共 {len(local_prompt.PROMPTS)} 个")


async def main():
    """主函数"""
    global MINERU_DEVICE
    
    parser = argparse.ArgumentParser(description="研报处理器")
    parser.add_argument("file_path", nargs='?', help="要处理的PDF或文本文件路径")
    parser.add_argument(
        "--working_dir", "-w", 
        default=WORKING_DIR, 
        help="工作目录路径"
    )
    parser.add_argument(
        "--query", "-q",
        help="要查询的问题"
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="检查依赖项安装状态"
    )
    parser.add_argument(
        "--device",
        default=MINERU_DEVICE,
        help=f"MinerU 处理设备 (默认: {MINERU_DEVICE})"
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="强制重新处理文档，忽略已存在的数据"
    )
    
    args = parser.parse_args()
    
    # 配置日志
    configure_logging()
    
    print("=" * 50)
    print("🏦 研报处理器")
    print("=" * 50)
    
    # 如果只是检查依赖项
    if args.check_deps:
        print("\n🔍 依赖项检查:")
        print(f"✅ RAGAnything: {'已安装' if HAS_RAGANYTHING else '未安装'}")
        print(f"✅ MinerU 2.0: {'已安装' if HAS_MINERU else '未安装'}")
        return
    
    # 检查是否提供了文件路径
    if not args.file_path:
        # 如果没有提供文件路径，检查是否有已存在的数据
        print("📋 未提供文件路径，检查是否有已处理的数据...")

        # 创建临时处理器实例来检查数据
        temp_processor = FinancialReportProcessor(args.working_dir)
        has_existing_data = await temp_processor.check_existing_data()

        if has_existing_data:
            print("✅ 发现已处理的数据！")
            print("🚀 检查并增量处理新文件...")

            # 使用已存在的数据初始化处理器
            processor = temp_processor
            try:
                await processor.load_existing_data()
                # 自动增量处理新文件
                update_results = await processor.update_knowledge_base_incrementally()
                if update_results:
                    new_count = sum(1 for v in update_results.values() if v)
                    print(f"📦 已增量处理/加载 {new_count} 个文件到知识库。")
                else:
                    print("📦 没有发现需要增量处理的新文件。")
                print("✅ 数据加载完成，可以开始查询")

                # 直接进入交互式查询模式
                print("\n💬 进入交互模式 (输入 'quit' 退出):")
                print("💡 提示: 您可以询问关于文档内容的任何问题")
                while True:
                    try:
                        question = input("\n请输入问题: ").strip()
                        if question.lower() in ['quit', 'exit', '退出']:
                            break

                        if question:
                            try:
                                response = await processor.query_document(question)
                                print(f"\n{response}")
                            except Exception as query_error:
                                print(f"查询失败: {query_error}")

                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        print(f"查询出错: {e}")

            except Exception as e:
                print(f"❌ 数据加载失败: {e}")
            finally:
                await processor.finalize()
                print("\n👋 程序结束")
                return
        else:
            print("❌ 错误: 请提供要处理的文件路径")
            print("使用 --help 查看完整帮助信息")
            print("\n💡 示例用法:")
            print("python financial_report_processor.py document.pdf")
            print("python financial_report_processor.py --check-deps  # 检查依赖项")
            print("python financial_report_processor.py  # 直接查询已处理的数据（如果存在）")
            print("python financial_report_processor.py document.pdf --force-reprocess  # 强制重新处理")
            return
    
    # 检查依赖项
    missing_deps = []
    if not HAS_RAGANYTHING:
        missing_deps.append("RAGAnything")
    if not HAS_MINERU:
        missing_deps.append("MinerU 2.0")
    
    if missing_deps:
        print(f"\n⚠️  警告: 缺少必要依赖项: {', '.join(missing_deps)}")
        print("某些功能将不可用")
        response = input("是否继续? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            return
    
    # 更新设备配置
    MINERU_DEVICE = args.device
    os.environ["MINERU_DEVICE"] = MINERU_DEVICE
    
    # 初始化处理器
    processor = FinancialReportProcessor(args.working_dir)
    
    try:
        # 初始化系统
        await processor.initialize()
        
        # 首先检查是否已有处理好的数据（除非强制重新处理）
        has_existing_data = False
        if not args.force_reprocess:
            has_existing_data = await processor.check_existing_data()
        
        if has_existing_data:
            print("✅ 发现已处理的数据，跳过文档处理步骤")
            print("🚀 直接进入查询模式...")
            print("💡 提示: 如需重新处理文档，请使用 --force-reprocess 参数")
            
            # 重新初始化以加载已有数据（不清理）
            await processor.load_existing_data()
        else:
            if args.force_reprocess:
                print("🔄 强制重新处理模式，将清理已有数据")
            
            # 处理文档
            print(f"\n📄 正在处理文档: {args.file_path}")
            success = await processor.process_document(args.file_path)
            
            if not success:
                print("❌ 文档处理失败")
                return
            
            print("✅ 文档处理完成")
        
        # 如果提供了查询问题，直接查询
        if args.query:
            print(f"\n💭 查询问题: {args.query}")
            print("� 使用 hybrid 模式和流式输出")
            
            try:
                response = await processor.query_document(args.query)
                print(f"\n{response}")
            except Exception as query_error:
                print(f"查询失败: {query_error}")
        else:
            # 交互式查询
            print("\n💬 进入交互模式 (输入 'quit' 退出):")
            print("💡 提示: 您可以询问关于文档内容的任何问题")
            while True:
                try:
                    question = input("\n请输入问题: ").strip()
                    if question.lower() in ['quit', 'exit', '退出']:
                        break
                    
                    if question:
                        try:
                            response = await processor.query_document(question)
                            print(f"\n{response}")
                        except Exception as query_error:
                            print(f"查询失败: {query_error}")
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"查询出错: {e}")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        print(f"❌ 程序执行失败: {e}")
    finally:
        # 只做资源清理，不做文件清理
        await processor.finalize()
        print("\n👋 程序结束")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
    # 主流程结束后单独同步清理 financial_output 目录
    try:
        # 重新实例化处理器以确保有working_dir
        processor = FinancialReportProcessor()
        processor.cleanup_all_output_files()
        print("\n🧹 已自动清理 financial_output 目录的无用文件")
    except Exception as e:
        print(f"\n⚠️ 自动清理 financial_output 目录失败: {e}")
