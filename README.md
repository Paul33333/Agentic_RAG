# DeepSearch复现篇_2（优势：低门槛）：基于DeepSeek官网API的Agentic RAG的全链路实现

全文阅读约2分钟~

## 背景

在上一篇[DeepSearch复现篇：QwQ-32B ToolCall功能初探，以Agentic RAG为例](https://zhuanlan.zhihu.com/p/30289363967)中，我们探讨了基于本地部署的**QwQ-32B**的**Agentic RAG**实现方案。然而实际工程落地中，云端API调用具有更低的准入和复现门槛。本文将以 **DeepSeek-R1官网API** 和 **Tavily检索API** 为核心（是的，你只需要申请这两个API的Key即可！），展示无需本地算力的全链路Agentic RAG实现方案。

> 关键优势：零本地部署，仅需两个API密钥，即可在本地实现端到端的**深度检索智能体**搭建！

相较于[DeepSearch复现篇：QwQ-32B ToolCall功能初探，以Agentic RAG为例](https://zhuanlan.zhihu.com/p/30289363967)的本地部署方案，本次升级的核心变化就是：

**模型层**：使用DeepSeek官网的的[DeepSeek-R1](https://platform.deepseek.com/api-docs)接口替代本地部署模型

---

## 详细实践

项目地址：[Agentic_RAG](https://github.com/Paul33333/Agentic_RAG)

- 1、下载项目地址文件：

```
git clone https://github.com/Paul33333/Agentic_RAG.git
```

- 2、安装必要的环境（tavily-python、dotenv、openai）

```
cd Agentic_RAG
pip install -r requirements.txt
```

- 3、设置API Key:
  - 将你的API KEY写进环境变量，避免显式暴露（非必须，你也可以改为不访问环境变量的方式，直接显式读取）

```
export DEEPSEEK_API_KEY="<Your DeepSeek API Key>"
export TAVILY_API_KEY="<Your Tavily API Key>"
```

- 4、运行程序

```
python Agentic_RAG.py
```

---

## 效果测试

一、**单轮问答**

输入问题：“**请给我详细介绍下前段时间小米发布的Su7 Ultra这款跑车的相关信息，要求信息尽可能全面**”

> 对比上一篇[DeepSearch复现篇：QwQ-32B ToolCall功能初探，以Agentic RAG为例](https://zhuanlan.zhihu.com/p/30289363967)中我们没有实现的多轮检索问答，这次我们成功触发了**多轮**（层次递进的）的联网搜索

返回结果具体如下：

其整体响应流程可以总结如下：

- 1、拿到问题-> 调用推理模型处理...
- 2、思考后，**触发第一次检索**：“**小米SU7 Ultra 正式发布 详细参数**”
- 3、读取检索响应后（可能还有思考），**其自身触发第二次检索**：“**小米SU7 Ultra 技术亮点**”
- 4、读取检索响应后（可能还有思考），**其自身触发第三次检索**：“**小米SU7 Ultra 市场反响**”
- 5、最后，结合三次检索的内容，**生成最后的输出**

> 这三次检索的流程编排、检索关键词设置均由模型动态指导自己完成

详细响应部分截图如下：
![截图 2025-03-18 16-36-02.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2025-03-18/1742287482765.png?raw=true)
![截图 2025-03-18 16-36-33.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2025-03-18/1742288296422.png?raw=true)
![截图 2025-03-18 16-37-28.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2025-03-18/1742288081577.png?raw=true)

二、**多轮问答**

在完成上面的第一个对话后，我们续着进行第二轮问答。

问题："**请你再继续对比下tesla modelS的产品信息， 给我生成一份小米Su7 Ultra和 tesla modelS的产品对比手册**"

详细响应部分截图如下（继续触发了两次检索）：

![截图 2025-03-18 17-07-40.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2025-03-18/1742289111403.png?raw=true)

![截图 2025-03-18 17-08-02.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2025-03-18/1742289615932.png?raw=true)

![截图 2025-03-18 17-08-24.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2025-03-18/1742289169899.png?raw=true)

![截图 2025-03-18 17-08-53.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2025-03-18/1742289637502.png?raw=true)

