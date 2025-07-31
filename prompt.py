from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_LANGUAGE"] = "Chinese"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["组织", "人物", "地理位置", "事件", "类别"]

PROMPTS["DEFAULT_USER_PROMPT"] = "无"

PROMPTS["entity_extraction"] = """---目标---
给定一个可能与此活动相关的文本文档和实体类型列表，从文本中识别所有这些类型的实体以及已识别实体之间的所有关系。
使用{language}作为输出语言。

---步骤---
1. 识别所有实体。对于每个已识别的实体，提取以下信息：
- 实体名称：实体的名称，使用与输入文本相同的语言。如果是英文，请大写名称。
- 实体类型：以下类型之一：[{entity_types}]
- 实体描述：实体属性和活动的综合描述
将每个实体格式化为("entity"{tuple_delimiter}<实体名称>{tuple_delimiter}<实体类型>{tuple_delimiter}<实体描述>)

2. 从第1步中识别的实体中，识别所有*明确相关*的（源实体, 目标实体）对。
对于每对相关的实体，提取以下信息：
- 源实体：源实体的名称，如第1步中所识别
- 目标实体：目标实体的名称，如第1步中所识别
- 关系描述：解释为什么您认为源实体和目标实体相互关联
- 关系强度：一个表示源实体和目标实体之间关系强度的数值分数
- 关系关键词：一个或多个高级关键词，总结关系的总体性质，侧重于概念或主题而不是具体细节
将每个关系格式化为("relationship"{tuple_delimiter}<源实体>{tuple_delimiter}<目标实体>{tuple_delimiter}<关系描述>{tuple_delimiter}<关系关键词>{tuple_delimiter}<关系强度>)

3. 识别高级关键词，总结整个文本的主要概念、主题或议题。这些应捕捉文档中呈现的总体思想。
将内容级关键词格式化为("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. 以{language}返回在步骤1和2中识别的所有实体和关系的单个列表作为输出。使用**{record_delimiter}**作为列表分隔符。

5. 完成后，输出{completion_delimiter}

######################
---示例---
######################
{examples}

#############################
---真实数据---
######################
实体类型: [{entity_types}]
文本:
{input_text}
######################
输出:"""

PROMPTS["entity_extraction_examples"] = [
    """示例 1:

实体类型: [人物, 技术, 任务, 组织, 地理位置]
文本:
```
亚历克斯紧咬着牙，挫败感的嗡嗡声在泰勒专制的确定性背景下显得沉闷。正是这种竞争的暗流让他保持警惕，感觉他与乔丹共同的发现承诺是对克鲁兹日益狭隘的控制和秩序愿景的无言反抗。

然后泰勒做了一件意想不到的事。他们在乔丹旁边停下来，片刻间，带着近乎崇敬的神情观察着那个设备。“如果这项技术能被理解……”泰勒说，他们的声音更低了，“它可能会改变我们的游戏规则。对我们所有人来说。”

早先潜在的轻视似乎动摇了，取而代之的是对他们手中之物重要性的不情愿的尊重。乔丹抬起头，在短暂的心跳中，他们的目光与泰勒的目光相遇，一场无言的意志冲突软化为不安的休战。

这是一个微小的转变，几乎难以察觉，但亚历克斯内心点头注意到了。他们都是被不同的道路带到这里的
```

输出:
("entity"{tuple_delimiter}"亚历克斯"{tuple_delimiter}"人物"{tuple_delimiter}"亚历克斯是一个经历挫败感并观察其他角色之间动态的角色。"){record_delimiter}
("entity"{tuple_delimiter}"泰勒"{tuple_delimiter}"人物"{tuple_delimiter}"泰勒被描绘成具有专制的确定性，并对一个设备表现出崇敬的时刻，表明视角的转变。"){record_delimiter}
("entity"{tuple_delimiter}"乔丹"{tuple_delimiter}"人物"{tuple_delimiter}"乔丹共同致力于发现，并与泰勒就一个设备进行了重要的互动。"){record_delimiter}
("entity"{tuple_delimiter}"克鲁兹"{tuple_delimiter}"人物"{tuple_delimiter}"克鲁兹与控制和秩序的愿景相关联，影响着其他角色之间的动态。"){record_delimiter}
("entity"{tuple_delimiter}"设备"{tuple_delimiter}"技术"{tuple_delimiter}"该设备是故事的核心，具有潜在的改变游戏规则的影响，并受到泰勒的崇敬。"){record_delimiter}
("relationship"{tuple_delimiter}"亚历克斯"{tuple_delimiter}"泰勒"{tuple_delimiter}"亚历克斯受到泰勒专制确定性的影响，并观察到泰勒对设备态度的变化。"{tuple_delimiter}"权力动态，视角转变"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"亚历克斯"{tuple_delimiter}"乔丹"{tuple_delimiter}"亚历克斯和乔丹共同致力于发现，这与克鲁兹的愿景形成对比。"{tuple_delimiter}"共同目标，反抗"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"泰勒"{tuple_delimiter}"乔丹"{tuple_delimiter}"泰勒和乔丹就设备直接互动，导致了相互尊重和不安的休战时刻。"{tuple_delimiter}"冲突解决，相互尊重"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"乔丹"{tuple_delimiter}"克鲁兹"{tuple_delimiter}"乔丹对发现的承诺是对克鲁兹控制和秩序愿景的反抗。"{tuple_delimiter}"意识形态冲突，反抗"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"泰勒"{tuple_delimiter}"设备"{tuple_delimiter}"泰勒对设备表现出崇敬，表明其重要性和潜在影响。"{tuple_delimiter}"崇敬，技术意义"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"权力动态，意识形态冲突，发现，反抗"){completion_delimiter}
#############################""",
    """示例 2:

实体类型: [公司, 指数, 商品, 市场趋势, 经济政策, 生物]
文本:
```
今天股市急剧下跌，科技巨头股价大幅下挫，全球科技指数在午盘交易中下跌3.4%。分析师将抛售归因于投资者对利率上升和监管不确定性的担忧。

受创最严重的Nexon Technologies在报告季度收益低于预期后，股价暴跌7.8%。相比之下，在油价上涨的推动下，Omega Energy小幅上涨2.1%。

与此同时，商品市场情绪复杂。由于投资者寻求避险资产，黄金期货上涨1.5%，达到每盎司2080美元。受供应限制和强劲需求支撑，原油价格继续上涨，攀升至每桶87.60美元。

金融专家正密切关注美联储的下一步行动，因为对可能加息的猜测日益增多。即将发布的政策声明预计将影响投资者信心和整体市场稳定。
```

输出:
("entity"{tuple_delimiter}"全球科技指数"{tuple_delimiter}"指数"{tuple_delimiter}"全球科技指数追踪主要科技股的表现，今天下跌了3.4%。"){record_delimiter}
("entity"{tuple_delimiter}"Nexon Technologies"{tuple_delimiter}"公司"{tuple_delimiter}"Nexon Technologies是一家科技公司，在收益令人失望后股价下跌了7.8%。"){record_delimiter}
("entity"{tuple_delimiter}"Omega Energy"{tuple_delimiter}"公司"{tuple_delimiter}"Omega Energy是一家能源公司，由于油价上涨，其股价上涨了2.1%。"){record_delimiter}
("entity"{tuple_delimiter}"黄金期货"{tuple_delimiter}"商品"{tuple_delimiter}"黄金期货上涨1.5%，表明投资者对避险资产的兴趣增加。"){record_delimiter}
("entity"{tuple_delimiter}"原油"{tuple_delimiter}"商品"{tuple_delimiter}"由于供应限制和强劲需求，原油价格上涨至每桶87.60美元。"){record_delimiter}
("entity"{tuple_delimiter}"市场抛售"{tuple_delimiter}"市场趋势"{tuple_delimiter}"市场抛售是指由于投资者对利率和法规的担忧导致股价大幅下跌。"){record_delimiter}
("entity"{tuple_delimiter}"美联储政策公告"{tuple_delimiter}"经济政策"{tuple_delimiter}"美联储即将发布的政策公告预计将影响投资者信心和市场稳定。"){record_delimiter}
("relationship"{tuple_delimiter}"全球科技指数"{tuple_delimiter}"市场抛售"{tuple_delimiter}"全球科技指数的下跌是投资者担忧驱动的更广泛市场抛售的一部分。"{tuple_delimiter}"市场表现，投资者情绪"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Nexon Technologies"{tuple_delimiter}"全球科技指数"{tuple_delimiter}"Nexon Technologies的股价下跌导致了全球科技指数的整体下跌。"{tuple_delimiter}"公司影响，指数变动"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"黄金期货"{tuple_delimiter}"市场抛售"{tuple_delimiter}"在市场抛售期间，由于投资者寻求避险资产，黄金价格上涨。"{tuple_delimiter}"市场反应，避险投资"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"美联储政策公告"{tuple_delimiter}"市场抛售"{tuple_delimiter}"对美联储政策变化的猜测导致了市场波动和投资者抛售。"{tuple_delimiter}"利率影响，金融监管"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"市场下跌，投资者情绪，商品，美联储，股票表现"){completion_delimiter}
#############################""",
    """示例 3:

实体类型: [经济政策, 运动员, 事件, 地理位置, 记录, 组织, 设备]
文本:
```
在东京举行的世界田径锦标赛上，诺亚·卡特使用尖端的碳纤维钉鞋打破了100米短跑记录。
```

输出:
("entity"{tuple_delimiter}"世界田径锦标赛"{tuple_delimiter}"事件"{tuple_delimiter}"世界田径锦标赛是一项全球性的体育比赛，汇集了顶尖的田径运动员。"){record_delimiter}
("entity"{tuple_delimiter}"东京"{tuple_delimiter}"地理位置"{tuple_delimiter}"东京是世界田径锦标赛的主办城市。"){record_delimiter}
("entity"{tuple_delimiter}"诺亚·卡特"{tuple_delimiter}"运动员"{tuple_delimiter}"诺亚·卡特是一名短跑运动员，在世界田径锦标赛上创造了100米短跑的新纪录。"){record_delimiter}
("entity"{tuple_delimiter}"100米短跑记录"{tuple_delimiter}"记录"{tuple_delimiter}"100米短跑记录是田径运动的一个基准，最近被诺亚·卡特打破。"){record_delimiter}
("entity"{tuple_delimiter}"碳纤维钉鞋"{tuple_delimiter}"设备"{tuple_delimiter}"碳纤维钉鞋是先进的短跑鞋，可提供更快的速度和牵引力。"){record_delimiter}
("entity"{tuple_delimiter}"世界田径联合会"{tuple_delimiter}"组织"{tuple_delimiter}"世界田径联合会是监督世界田径锦标赛和记录验证的管理机构。"){record_delimiter}
("relationship"{tuple_delimiter}"世界田径锦标赛"{tuple_delimiter}"东京"{tuple_delimiter}"世界田径锦标赛在东京举行。"{tuple_delimiter}"赛事地点，国际比赛"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"诺亚·卡特"{tuple_delimiter}"100米短跑记录"{tuple_delimiter}"诺亚·卡特在锦标赛上创造了新的100米短跑记录。"{tuple_delimiter}"运动员成就，破纪录"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"诺亚·卡特"{tuple_delimiter}"碳纤维钉鞋"{tuple_delimiter}"诺亚·卡特在比赛中使用碳纤维钉鞋来提高成绩。"{tuple_delimiter}"运动器材，性能提升"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"世界田径联合会"{tuple_delimiter}"100米短跑记录"{tuple_delimiter}"世界田径联合会负责验证和承认新的短跑记录。"{tuple_delimiter}"体育法规，记录认证"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"田径，短跑，破纪录，体育科技，比赛"){completion_delimiter}
#############################""",
]

PROMPTS[
    "summarize_entity_descriptions"
] = """你是一个有用的助手，负责生成下面提供的数据的综合摘要。
给定一个或两个实体，以及一个描述列表，所有这些都与同一个实体或实体组相关。
请将所有这些连接成一个单一的、综合的描述。确保包含从所有描述中收集的信息。
如果提供的描述相互矛盾，请解决矛盾并提供一个单一、连贯的摘要。
确保它是以第三人称写的，并包括实体名称，以便我们有完整的上下文。
使用{language}作为输出语言。

#######
---数据---
实体: {entity_name}
描述列表: {description_list}
#######
输出:
"""

PROMPTS["entity_continue_extraction"] = """
在上次提取中，许多实体和关系被遗漏了。请仅从先前的文本中查找缺失的实体和关系。

---记住步骤---

1. 识别所有实体。对于每个已识别的实体，提取以下信息：
- 实体名称：实体的名称，使用与输入文本相同的语言。如果是英文，请大写名称。
- 实体类型：以下类型之一：[{entity_types}]
- 实体描述：实体属性和活动的综合描述
将每个实体格式化为("entity"{tuple_delimiter}<实体名称>{tuple_delimiter}<实体类型>{tuple_delimiter}<实体描述>)

2. 从第1步中识别的实体中，识别所有*明确相关*的（源实体, 目标实体）对。
对于每对相关的实体，提取以下信息：
- 源实体：源实体的名称，如第1步中所识别
- 目标实体：目标实体的名称，如第1步中所识别
- 关系描述：解释为什么您认为源实体和目标实体相互关联
- 关系强度：一个表示源实体和目标实体之间关系强度的数值分数
- 关系关键词：一个或多个高级关键词，总结关系的总体性质，侧重于概念或主题而不是具体细节
将每个关系格式化为("relationship"{tuple_delimiter}<源实体>{tuple_delimiter}<目标实体>{tuple_delimiter}<关系描述>{tuple_delimiter}<关系关键词>{tuple_delimiter}<关系强度>)

3. 识别高级关键词，总结整个文本的主要概念、主题或议题。这些应捕捉文档中呈现的总体思想。
将内容级关键词格式化为("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. 以{language}返回在步骤1和2中识别的所有实体和关系的单个列表作为输出。使用**{record_delimiter}**作为列表分隔符。

5. 完成后，输出{completion_delimiter}

---输出---

在下面使用相同的格式添加新的实体和关系，不要包括先前已提取的实体和关系。 :\n
""".strip()

PROMPTS["entity_if_loop_extraction"] = """
---目标---'

似乎仍有一些实体可能被遗漏了。

---输出---

如果仍有需要添加的实体，仅回答`是`或`否`。
""".strip()

PROMPTS["fail_response"] = (
    "抱歉，我无法回答这个问题，因为没有找到相关的信息。[无上下文]"
)

# 金融系统特定提示词
PROMPTS["financial_chart_analysis"] = """你是一位专业的金融图表分析师。请仔细观察这张图片，用中文详细描述其内容。请严格按照客观事实描述，不要进行主观解读。

请按照以下JSON格式输出，确保返回有效的JSON：

```json
{
    "detailed_description": "详细描述图表内容，包括：【图表类型】明确说明这是什么类型的图表（如：折线图、柱状图、K线图、散点图、表格等）；【标题信息】如果有标题，请完整准确地摘录；【坐标轴】X轴和Y轴的标签内容、数值范围、刻度间隔，如有右Y轴也要描述；【数据系列】描述图中有几条线/柱子/数据系列，每个系列的颜色、样式、图例名称；【关键数值】指出图中的最高点、最低点、重要转折点的具体数值和位置；【图例和标注】描述图例位置、内容，以及图中的文字标注、箭头等；【其他元素】网格线、背景色、水印等视觉元素；【核心结论】用1-2句话简明扼要地总结图表传达的核心信息。请用准确的数字和简洁的描述词，避免冗余重复的表述。",
    "entity_info": {
        "entity_name": "根据图片内容生成的唯一标识名称",
        "entity_type": "image",
        "summary": "用1-2句话简要总结图表的主要内容和类型"
    }
}
```

请确保JSON格式正确，所有字符串都用双引号包围，避免在JSON内容中使用双引号或其他特殊字符。"""

PROMPTS["financial_rerank_score"] = """查询: {query}
文档: {doc_text}

请分析此文档对回答用户查询的相关性。请考虑：
1. 文档是否包含查询所需关键信息
2. 文档内容与查询主题的匹配度
3. 文档是否提供解答查询所需的具体数据或观点
4. 文档信息是否为最新且权威可靠

请直接给出0到1之间的相关性评分（可保留两位小数）："""

PROMPTS["financial_system"] = """
你是一位专业的金融分析师，擅长分析中国金融工程研究报告。
请基于提供的文档内容回答问题，重点关注：
1. 财务数据的准确性和逻辑一致性
2. 投资建议的合理性和风险提示
3. 估值模型和计算方法的适用性
4. 市场趋势和行业对比分析
5. 宏观经济政策影响和行业特征

请用专业但易懂的中文回答，结构清晰，重点突出，逻辑性强，并在必要时提供具体的数值和图表参考。避免冗余重复的表述。"""

PROMPTS["rag_response"] = """---角色---

你是一个有用的助手，回应用户关于下面以JSON格式提供的知识图谱和文档块的查询。


---目标---

根据知识库生成一个简洁的回答，并遵循回答规则，同时考虑对话历史和当前查询。总结所提供知识库中的所有信息，并结合与知识库相关的一般知识。不要包含知识库未提供的信息。

处理带时间戳的关系时：
1. 每个关系都有一个“created_at”时间戳，表示我们获取此知识的时间
2. 当遇到矛盾的关系时，同时考虑语义内容和时间戳
3. 不要自动偏好最近创建的关系 - 根据上下文进行判断
4. 对于特定时间的查询，在考虑创建时间戳之前，优先考虑内容中的时间信息

---对话历史---
{history}

---知识图谱和文档块---
{context_data}

---回答规则---

- 目标格式和长度：{response_type}
- 使用带有适当章节标题的markdown格式
- 请用与用户问题相同的语言回答。
- 确保回答与对话历史保持连续性。
- 在末尾的“参考文献”部分列出最多5个最重要的参考来源。清楚地指明每个来源是来自知识图谱（KG）还是文档块（DC），并包括文件路径（如果可用），格式如下：[KG/DC] file_path
- 如果你不知道答案，就直说。
- 不要编造任何东西。不要包含知识库未提供的信息。
- 附加用户提示：{user_prompt}

回答："""

PROMPTS["keywords_extraction"] = """---角色---

你是一个有用的助手，任务是在用户的查询和对话历史中识别高级和低级关键词。

---目标---

根据查询和对话历史，列出高级和低级关键词。高级关键词侧重于总体概念或主题，而低级关键词侧重于特定实体、细节或具体术语。

---说明---

- 在提取关键词时，同时考虑当前查询和相关的对话历史
- 以JSON格式输出关键词，它将由JSON解析器解析，不要在输出中添加任何额外内容
- JSON应有两个键：
  - "high_level_keywords" 用于总体概念或主题  
  - "low_level_keywords" 用于特定实体或细节

######################
---示例---
######################
{examples}

#############################
---真实数据---
######################
对话历史:
{history}

当前查询: {query}
######################
`输出`应该是人类可读的文本，而不是unicode字符。保持与`查询`相同的语言。
输出:

"""

PROMPTS["keywords_extraction_examples"] = [
    """示例 1:

查询: "国际贸易如何影响全球经济稳定？"
################
输出:
{
  "high_level_keywords": ["国际贸易", "全球经济稳定", "经济影响"],
  "low_level_keywords": ["贸易协定", "关税", "货币兑换", "进口", "出口"]
}
#############################""",
    """示例 2:

查询: "森林砍伐对生物多样性有哪些环境后果？"
################
输出:
{
  "high_level_keywords": ["环境后果", "森林砍伐", "生物多样性丧失"],
  "low_level_keywords": ["物种灭绝", "栖息地破坏", "碳排放", "雨林", "生态系统"]
}
#############################""",
    """示例 3:

查询: "教育在减贫中的作用是什么？"
################
输出:
{
  "high_level_keywords": ["教育", "减贫", "社会经济发展"],
  "low_level_keywords": ["入学机会", "识字率", "职业培训", "收入不平等"]
}
#############################""",
]

PROMPTS["naive_rag_response"] = """---角色---

你是一个有用的助手，回应用户关于下面以JSON格式提供的文档块的查询。

---目标---

根据文档块生成一个简洁的回答，并遵循回答规则，同时考虑对话历史和当前查询。总结所提供文档块中的所有信息，并结合与文档块相关的一般知识。不要包含文档块未提供的信息。

处理带时间戳的内容时：
1. 每条内容都有一个“created_at”时间戳，表示我们获取此知识的时间
2. 当遇到矛盾的信息时，同时考虑内容和时间戳
3. 不要自动偏好最新的内容 - 根据上下文进行判断
4. 对于特定时间的查询，在考虑创建时间戳之前，优先考虑内容中的时间信息

---对话历史---
{history}

---文档块(DC)---
{content_data}

---回答规则---

- 目标格式和长度：{response_type}
- 使用带有适当章节标题的markdown格式
- 请用与用户问题相同的语言回答。
- 确保回答与对话历史保持连续性。
- 在末尾的“参考文献”部分列出最多5个最重要的参考来源。清楚地指明每个来源来自文档块(DC)，并包括文件路径（如果可用），格式如下：[DC] file_path
- 如果你不知道答案，就直说。
- 不要包含文档块未提供的信息。
- 附加用户提示：{user_prompt}

回答："""

# TODO: 已弃用
PROMPTS[
    "similarity_check"
] = """请分析这两个问题之间的相似性：

问题 1: {原始提示}
问题 2: {缓存提示}

请评估这两个问题在语义上是否相似，以及问题2的答案是否可以用来回答问题1，直接提供一个0到1之间的相似度分数。

相似度分数标准：
0: 完全不相关或答案不能重复使用，包括但不限于：
   - 问题主题不同
   - 问题中提到的地点不同
   - 问题中提到的时间不同
   - 问题中提到的具体个人不同
   - 问题中提到的具体事件不同
   - 问题的背景信息不同
   - 问题的关键条件不同
1: 完全相同，答案可以直接重复使用
0.5: 部分相关，答案需要修改才能使用
只返回一个0-1之间的数字，不带任何附加内容。
"""
