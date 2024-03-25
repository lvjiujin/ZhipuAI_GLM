import time
from dotenv import load_dotenv
load_dotenv()

from api import get_characterglm_response, generate_character_profile

import gradio as gr


def gen_role_character_profile(text1, text2):
    s1 = "".join(generate_character_profile(text=text1))
    s2 = "".join(generate_character_profile(text=text2))
    # print("s1 = ", s1)
    s1 = s1[8:-3].strip("/n").strip()
    s2 = s2[8:-3].strip("/n").strip()
    import json
    d1 = json.loads(s1)
    d2 = json.loads(s2)
    role1 = d1['role']
    character_profile1 = d1['character_profile']
    role2 = d2['role']
    character_profile2 = d2['character_profile']

    return role1, character_profile1, role2, character_profile2

with gr.Blocks() as demo:
    txt1 = gr.Textbox(label="人物1", lines=2)
    txt2 = gr.Textbox(label="人物2")
    role1 = gr.Textbox(value="",label="角色1")
    pf1 = gr.Textbox(value="", label="人设1")
    role2 = gr.Textbox(value="",label="角色2")
    pf2 = gr.Textbox(value="", label="人设2")
    btn = gr.Button(value="Submit")
    btn.click(gen_role_character_profile, inputs=[txt1,txt2], outputs=[role1, pf1, role2,  pf2])
demo.launch()





    
