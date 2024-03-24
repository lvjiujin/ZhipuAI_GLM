import time
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
        {"role": "assistant", "content": "哥哥，我会死吗？"},
        {"role": "user", "content": "（微信）怎么会呢？医生说你的病情已经好转了"}
    ]
    for chunk in get_characterglm_response(messages, meta=character_meta):
        print(chunk)
        time.sleep(0.5)





if __name__ == "__main__":
    # characterglm_example()
    text1 = """
JoJean Retrum
JoJean Retrum is the Director and owner of Monona Academy of Dance and La Boutique Dance Wear.  Ms. Retrum’s training includes studying at American Ballet Theater, Harkness House in NYC and with teachers such as Jean Adams, Gus Giordano, and Luigi and Joe Tremaine.  She is a graduate of UW-Milwaukee in dance, music and theater.  She danced professional with Milwaukee Ballet, Ruth Page Company and Melody Top Theater.

Ms. Retrum is known for her ability to teach technique and is often a sought after guest instructor in her specialty.  She recently completed a guest educator position with the UW-Madison Dance Department teaching technique.  She has been honored by the Dance Council of Wisconsin for her achievements in dance, as the Outstanding Teacher in 2002 by Youth America Grand Prix and served as Regional Dance America’s - MidStates President 2012-2014.

Ms. Retrum is dedicated to creating an atmosphere which fosters self-esteem, discipline and positive influences; all of which can be carried into other aspects of the dance student’s life. Her dream is that everyone would love dance and participate.  As the founder and Artistic Director of Dance Wisconsin she provides performance opportunities for dancers all over southern and central Wisconsin.  As the energy and organizer of Dance Wisconsin’s Dance in the Schools Program she brings dance to student’s in their physical education classes with two and three week residencies.  These grant funded residencies bring dance into schools with free and reduced meal programs near or above 50% to begin the life long interest in dance.

Ms. Retrum is extremely proud of her dancers that have gone on to professional careers.  They are numerous.  She is also very proud of the dancers that have used the work ethic learned from dance to propel themselves into other amazing careers such as lawyers, doctors, nurses, choreographers, teachers, mothers, fathers and more.  
"""
    text2 = """
吕不韦（？—前235年），姜姓，吕氏。中国战国时代政治人物，卫国濮阳（今河南濮阳南）人[注 1]，初为大商人，后来成为秦相，封文信侯，在秦为相十三年。广招门客以“兼儒墨，合名法”为思想中心，合力编撰《吕氏春秋》，有系统性的提出自己的政治主张，后为先秦杂家代表人物之一[1]。
执政时曾攻取周、赵、卫的土地，立三川、太原、东郡，对秦王政兼并六国的事业有重大贡献。后因嫪毐集团叛乱事受牵连，被免除相邦职务，出居河南封地。不久，秦王政下令将其流放至蜀（今四川），不韦忧惧交加，于是在三川郡（今河南洛阳）自鸩而亡。
"""
    text3 = """
秦始皇（前259年2月18日—前210年7月11日[参1]），嬴姓，赵氏，名政，时称赵政（或称赵正），史书多作秦王政或始皇帝。 祖籍嬴城（今山东济南市莱芜区）[参2][参3][参4][参5]，生于赵国首都邯郸（今河北邯郸市），是秦庄襄王及赵姬之子[古4]，商朝重臣恶来的第35世孙。出土《北京大学藏西汉竹书》第三卷中称其为赵正。唐代司马贞在《史记索隐》引述《世本》称其为赵政[注4][参6]。曹植《文帝诔》最早称始皇帝为嬴政[参7]，后世通称嬴政，亦被某些文学作品称为“祖龙”[注5]。他是战国末期秦国君主，十三岁即位，先后铲除嫪毐与吕不韦，并重用李斯、尉缭，三十九岁时灭亡六国建立秦朝，自称“始皇帝”，五十岁出巡时驾崩[参8]，在位三十七年。
秦始皇是中国史上第一位使用“皇帝”称号的君主[参9]。统一天下后，秦始皇继承了商鞅变法的郡县制度和中央集权[参9]，统一度量衡，“车同轨，书同文，行同伦[参10]”及典章法制[参8]，奠定了中国政治史上两千馀年之专制政治格局，他被明代思想家李贽誉为“千古一帝”。但另一方面，秦始皇在位期间亦进行多项大型工程，包括修筑长城、阿房宫、骊山陵等，施政急躁，令人民徭役过重，是秦朝在他死后3年迅速灭亡的重要原因。
"""

    s = "".join(generate_character_profile(text=text1))
    s = s[8:-3].strip("/n").strip()
    import json
    d = json.loads(s)
    print(type(d))
    print(d['role'])
    print(d['character_profile'])
    
    
    

