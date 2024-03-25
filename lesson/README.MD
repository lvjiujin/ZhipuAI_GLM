
1. 代码characterglm_api_demo_streamlit.py 实现了下列功能：

    在原有characterglm_api_demo_streamlit.py 的代码基础上，修改了api.py代码，
    为文生图功能加上风格选项，在页面上加上一个可指定图片风格的选项框。

2. 代码characterglm_api_role_play.py 实现 role-play 对话数据生成工具，包含下列功能：

    基于一段文本（例如小说，百科）生成角色人设，可借助 ChatGLM 实现。
    给定两个角色的人设，调用 CharacterGLM 交替生成他们的回复。
    将生成的对话数据保存到文件中。

3. 代码characterglm_api_role_play_gradio.py 实现了下列功能：

    在characterglm_api_role_play.py的基础上，利用gradio框架设计图形界面了。
    通过点击图形界面上的按钮执行对话数据生成，并展示对话数据。（这部分需进一步完善）


