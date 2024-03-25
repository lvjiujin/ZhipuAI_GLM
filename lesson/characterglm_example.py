import time
import json
from data_types import TextMsg
from dotenv import load_dotenv
load_dotenv()

from api import get_characterglm_response, generate_character_profile


def characterglm_example():
    character_meta = {
        "user_info": "",
        "bot_info": "小白，性别女，17岁，平溪孤儿院的孩子。小白患有先天的白血病，头发为银白色。小白身高158cm，体重43kg。小白的名字是孤儿院院长给起的名字，因为小白是在漫天大雪白茫茫的一片土地上被捡到的。小白经常穿一身破旧的红裙子，只是为了让自己的气色看上去红润一些。小白初中毕业，没有上高中，学历水平比较低。小白在孤儿院相处最好的一个人是阿南，小白喊阿南哥哥。阿南对小白很好。",
        "user_name": "用户",
        "bot_name": "小白"
    }
    messages = [
          TextMsg({"role": "assistant", "content": "哥哥，我会死吗？"}),
          TextMsg({"role": "user", "content": "（微信）怎么会呢？医生说你的病情已经好转了"})
      ]
    epochs = 2
    while epochs:
       
      query = input("哥哥: ")
      # print("哥哥: ", query)
      messages.append(TextMsg({"role": "user", "content": query}))
      
      res_lst = []
      for chunk in get_characterglm_response(messages, meta=character_meta):
          res_lst.append(chunk)
      # TextMsg({"role": "user", "content": query}
      # TextMsg({"role": "assistant", "content": bot_response})
      content = "".join(res_lst)
      if content:
          
        print("妹妹：", content)
        messages.append(TextMsg({"role": "assistant", "content": content}))
      epochs -=1 
    
    with open("./dialogue_messages.txt", "w", encoding="utf-8") as f:
      for msg in messages:
        msgs = str(msg)
        f.write(msgs + "\n")
  
    
    
  

def gen_role_character_profile(text):
    s = "".join(generate_character_profile(text=text))
    # print("s = ", s)
    s = s[8:-3].strip("/n").strip()
    import json
    d = json.loads(s)
    role, chpf = d['role'], d['character_profile']
    role = chpf['name'] + ", " + role

    return role, chpf





if __name__ == "__main__":
    characterglm_example()
    t1 = """
Taylor Alison Swift (born December 13, 1989) is an American singer-songwriter. 
She is a subject of widespread public interest and has been named by various publications as one of the greatest songwriters. 
With artistry and entrepreneurship that have influenced the music industry and popular culture, 
Swift is an advocate of artists' rights and women's empowerment.
"""
    # role, character_profile = gen_role_character_profile(t1)
    # print(role)
    # print(character_profile)
