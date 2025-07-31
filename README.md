# EastmoneyReportRAG

## 项目简介
EastmoneyReportRAG 是一个面向金融研报的自动化采集、处理、知识抽取与增强检索（RAG）系统。项目集成了爬虫、PDF 批量解析、图像理解、知识图谱构建与大模型问答等能力，适用于金融行业的研报数据处理和智能问答场景。

## 主要功能
- 自动爬取东方财富网宏观/策略研报 PDF
- 批量解析 PDF，抽取文本与图像，图像自动生成中文描述
- 基于 MinerU、LightRAG、Ollama 等多模态大模型进行知识抽取与问答
- 构建和维护本地知识库与知识图谱
- 支持断点续传、进度追踪、日志管理
- 结构化输出 Markdown 文件，便于后续分析与展示

## 目录结构
```
financial_report_processor.py      # 主处理与RAG流程
pdf_processor.py                  # PDF批量解析与图像描述
prompt.py                         # 配合financial_report_processor.py使用的大模型提示词
Crawler/
    MacroReportCrawler.py         # 宏观研报爬虫
    StrategyReportCrawler.py      # 策略研报爬虫
financial_output/                 # 结构化输出结果
financial_rag/knowledge_base/     # 知识库与图谱数据
logs/                             # 日志
```

## 快速开始
1. **环境准备**
   - Python 3.9 及以上
   - 推荐使用虚拟环境（如 mamba）
2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```
3. **配置 MinerU/LightRAG/Ollama 环境**
   - 需提前安装并配置好 [MinerU](https://github.com/OpenDataLab/MinerU)、[LightRAG](https://github.com/HKUDS/LightRAG) 及 Ollama 本地大模型服务
   - mineru.json 配置文件需放在用户主目录或项目根目录
4. **采集研报 PDF**
   ```bash
   python Crawler/MacroReportCrawler.py
   python Crawler/StrategyReportCrawler.py
   ```
5. **批量解析 PDF 并生成 Markdown（只进行解析，独立于financial_report_processor.py）**
   ```bash
   python pdf_processor.py
   ```
6. **PDF批量解析+知识抽取与问答（RAG）主流程**
   ```bash
   python financial_report_processor.py
   ```

## 依赖说明
- requests, tqdm, beautifulsoup4：爬虫与下载
- mineru, lightrag, raganything：大模型与RAG核心
- loguru, python-dotenv：日志与环境变量
- 其他依赖详见 requirements.txt

## 进阶用法
- 支持自定义模型、知识库路径、输出格式等高级参数
- 可扩展接入更多大模型与知识图谱工具
