import time
import json
from data_types import TextMsg
from dotenv import load_dotenv
load_dotenv()

from api import get_characterglm_response, generate_character_profile


def characterglm_chat(role1, chpf1, role2, chpf2):
    character_meta = {
       "user_name": str(role1),
       "bot_name":  str(role2),
        "user_info": str(chpf1),
        "bot_info": str(chpf2)
    }
    messages = [
        TextMsg({"role": "user", "content": "杰伦好呀，很喜欢听你的中文歌"}),
        TextMsg({"role": "assitant", "content": "霉霉好，听说你要在悉尼开演唱会？"})]
    epochs = 3
    while epochs:
       
      query = input(role1['name'] + ": ")
      
      messages.append(TextMsg({"role": "user", "content": query}))
      
      res_lst = []
      for chunk in get_characterglm_response(messages, meta=character_meta):
          res_lst.append(chunk)
   
      content = "".join(res_lst)
    #   print("content = ", content)
      if content:
          
        print(role2["name"] + "：", content)
        messages.append(TextMsg({"role": "assistant", "content": content}))
      epochs -=1 
    
    with open("./dialogue_messages.txt", "w", encoding="utf-8") as f:
      for msg in messages:
        msgs = str(msg)
        f.write(msgs + "\n")
  
    
    
  

def gen_role_character_profile(text):
    s = "".join(generate_character_profile(text=text))
    if s.startswith("```plaintext"):
       s = s[12:-3].strip("/n").strip()
    
    else:
        s = s[8:-3].strip("/n").strip()

    d = json.loads(s)
    role, character_profile = d['role'], d['character_profile']
    

    return role, character_profile





if __name__ == "__main__":
    t1 = """
泰勒·艾莉森·斯威夫特（英语：Taylor Alison Swift；1989年12月13日—），美国创作歌手及音乐制作人。
斯威夫特的音乐作品跨越了多种类型，她还会基于个人生活创作歌曲。
斯威夫特出生于宾夕法尼亚州西雷丁，为了追求乡村音乐事业，14岁时举家搬至田纳西州纳什维尔。
她在2004年与索尼/联合电视音乐出版签订了歌曲创作协议，在2005年与大机器唱片签订了唱片合约，并在2006年发行了她同名的首张录音室专辑。
"""
    t2 = """
周杰伦（英语：Jay Chou；1979年1月18日—），台湾创作男歌手、钢琴家、词曲作家、唱片制片人。
2000年周杰伦推出了首张专辑《杰伦》，他的音乐遍及亚太区和西方国家的华人社群，是华语乐坛极具影响力的音乐人，有“亚洲流行天王”之美誉，并获得了多个重要音乐奖项，包括15座台湾金曲奖和2座MTV亚洲大奖，周杰伦也为其他歌手写歌。2003年他登上亚洲版《时代》杂志的封面，其后开展了六个世界巡演。 2009年获美国CNN评选为亚洲最具影响力的二十五位人物之一。[2]2022年推出的专辑 《最伟大的作品》 ，夺下IFPI认证全球最畅销专辑冠军。
"""
    role1, chpf1 = gen_role_character_profile(t1)
    role2, chpf2 = gen_role_character_profile(t2)
    # print(type(role1))
    # print(type(chpf1))
    characterglm_chat(role1, chpf1, role2, chpf2)

    
