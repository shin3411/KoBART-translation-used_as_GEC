from tqdm import tqdm_notebook

import pandas as pd
import numpy as np
import re
import shutil
import random
from g2pk import G2p
import os

nli_train_csv_path = os.path.dirname(os.path.realpath(__file__)) + "/nli.train.ko.csv"
nli_complete_train_csv_path_4 = os.path.dirname(os.path.realpath(__file__)) + "/nli.train_compelete.ko_4.csv"
nli_complete_train_csv_path_5 = os.path.dirname(os.path.realpath(__file__)) + "/nli.train_compelete.ko_5.csv"

print(nli_train_csv_path)

random.seed() # 추가
# random.seed(a=None, version=2)
# 난수 생성기를 초기화합니다.
# a가 생략되거나 None이면, 현재 시스템 시간이 사용됩니다. 

g2p = G2p()

confusing_dict = {
    '마음': ['맴', '맘'],
    '세다': ['쎄다'],
    '안에는': ['안엔'],
    '밖에는': ['밖엔'],
    '귀여운': ['뽀짝'],
    '이런': ['요런'],
    '시간': ['time', '타임'],
    '주스': ['쥬스'],
    '아': ['애'],
    '돼': ['되'],
    '되': ['돼'],
    '된': ['됀'],
    '안': ['않'],
    '않': ['안'],
    '왜': ['외'],
    '원': ['one'],
    '가격': ['price'],
    '사이즈': ['싸이즈'],
    '주문': ['오더', 'order'],
    '대박': ['대애박', '대애애박', '대애애애박'],
    '피시': ['피', '피씨', 'PC'],
    '바람': ['바램'],
    '너무': ['넘', '늠', '느무', '느무느무'],
    '좋아': ['죠아', '조아', '조아조아'],
    '바탕': ['베이스'],
    '아기': ['애기', '애긔'],
    '엄마': ['맘', '마미'],
    '어머니': ['맘', '마미'],
    '음악': ['뮤직'],
    '갔다': ['가따'],
    '왔다': ['와따'],
    '설레': ['설레이'],
    '버렸어': ['부렀어', '부럿어'],
    '딱': ['뙇'],
    '그저 그렇': ['쏘쏘', '소소'],
    '그저 그러': ['쏘쏘', '소소'],
    '그저 그런': ['쏘쏘', '소소'],
    '이렇': ['요렇', '요롷'],
    '갑니다': ['고고', 'gogo'],
    '가십시오': ['고고', 'gogo', '고고씽', '고고링'],
    '고기': ['꼬기'],
    '좋은': ['굿', '굳'],
    '좋다': ['굿', '굳'],
    '하루': ['데이'],
    '하나도': ['1도'],
    '메뉴': ['menu'],
    ' 대 ': [' vs '],
    '팩스': ['fax'],
    '이메일': ['email', 'e-mail'],
    '진짜': ['실화', '리얼'],
    '놀람': ['놀램'],
    '채팅': ['챗'],
    '최고의': ['베스트'],
    '맛있는 것': ['jmt', '존맛탱', '존맛', '킹맛'],
    '맛있다': ['jmt', '존맛탱', '존맛', '킹맛'],
    '고추냉이': ['와사비'],
    '전망': ['뷰'],
    '시작': ['스타트', '스따또', '스타또', '스타뜨'],
    '이건': ['요건'],
    '어떻게': ['어케'],
    '스테이크': ['steak', '스떼끼', '스테끼'],
    '누구를': ['누굴'],
    '재미있다': ['존잼', '잼'],
    '요가': ['yoga'],
    '때는': ['땐'],
    '바로': ['right', '롸잇'],
    '예약': ['reservation'],
    '되게': ['디게', '디기'],
    '여기': ['요기'],
    '저기': ['죠기', '조기'],
    '구웠': ['굽굽'],
    '고양이': ['냥이', '냥냥이', '냥', '고냥이'],
    '하지만': ['but'],
    '꾸덕꾸덕': ['꾸덕'],
    '잖아': ['쟈나', '쟈냐'],
    '얼른': ['언능', '언넝', '얼릉', '얼렁'],
    '정말': ['겁나', '검나', '존나'],
    '주차': ['파킹', 'parking'],
    '예뻐': ['예뿌', '이뿌'],
    '잘 어울리': ['찰떡'],
    '짭조름': ['짬조롬', '짬쪼롬', '짭자름'],
    '작은': ['쪼꼼', '쪼꼬미'],
    '행사': ['이벤트'],
    '수준': ['퀄리티', '레벨'],
    '후기': ['리뷰'],
    '힘': ['파워풀'],
    '치명적인': ['치명치명한'],
    '기다리': ['웨이팅하'],
    '죠': ['쥬'],
    '최고': ['굿', '굳'],
    '귀엽다': ['귀염', '귀욥', '귀욤'],
    '맛있': ['마싯'],
    '어이': ['어의'],
    '검은': ['검정'],
    '나았': ['낳았'],
    '낫': ['났', '낳'],
    '일부러': ['일부로'],
    '함부로': ['함부러'],
    '가르치': ['가리키'],
    '가리키': ['가르치'],
    '무난': ['문안'],
    '오랜만': ['오랫만'],
    '얘기': ['예기'],
    '금세': ['금새'],
    '웬': ['왠'],
    '왠': ['웬'],
    '며칠': ['몇일', '몇 일'],
    '교향곡': ['교양곡'],
    '훼손': ['회손'],
    '사달': ['사단'],
    '머리말': ['머릿말'],
    '기를': ['길'],
    '어떻': ['어떡'],
    '어떡': ['어떻'],
    '피우': ['피'],
    '불리': ['불리우'],
    '게': ['개'],
    '끗발': ['끝발'],
    '는지': ['런지'],
    '대갚음': ['되갚음'],
    '띄어쓰기': ['띄워쓰기'],
    '목말': ['목마'],
    '범칙': ['벌칙'],
    '희한': ['희안'],
    '감금': ['강금'],
    '꿰': ['꼬'],
    '됐': ['됬'],
    '몹쓸': ['못쓸'],
    '으레': ['으례', '의례', '의레'],
    '이따가': ['있다가'],
    '담백': ['단백'],
    '쩨쩨': ['째째'],
    '굽실': ['굽신'],
    '비비': ['부비'],
    '악천후': ['악천우'],
    '결재': ['결제'],
    '결제': ['결재'],
    '지양': ['지향'],
    '지향': ['지양'],
    '깎': ['깍'],
    '연루': ['연류'],
    '덮밥': ['덥밥'],
    '결딴': ['결단'],
    '늘그막': ['늙으막'],
    '고진감래': ['고진감내'],
    '사면초가': ['사면초과'],
    '환골탈태': ['환골탈퇴'],
    '어폐': ['어패'],
    '바치': ['받치'],
    '해코지': ['해꼬지'],
    '뭐': ['모', '머'],
    '뭘': ['몰', '멀'],
    '뭔': ['먼', '몬'],
    '뭐를': ['멀', '몰'],
    '가게': ['숍', '샵'],
    '감사': ['ㄳ', 'ㄱㅅ'],
    '수고': ['ㅅㄱ'],
    # 추가
    '저난도': ['저난이도'],
    '고난도': ['고난이도'],
    '어렵게': ['높은 난이도로', '고난이도로'],
    '쉽게': ['낮은 난이도로', '저난이도로'],
    '날아가': ['날라가'],
    '날짜': ['날자'],
    '납량': ['남량', '남냥'],
    '나지막': ['나즈막'],
    '내로라': ['내노라','내놓으라'],
    '네가': ['너가', '니가'],
    '네댓 ': ['너댓 '],
    '너덧 ': ['너댓 '],
    '너비': ['넓이'],
    '넓이': ['너비'],
    '널브러지다': ['널부러지다'],
    '멀리뛰기': ['넓이뛰기'],
    '저녁': ['저녘'],
    '저녁녘': ['저녘녘'],
    '놈팡이': ['놈팽이'],
    '누다': ['싸다'],
    '누누이': ['누누히'],
    '눈곱': ['눈꼽'],
    '눈살': ['눈쌀'],
    '눋는': ['눌는'],
    '눌은': ['눋은'],
    '느냐고': ['느라고'],
    '느라고': ['느냐고'],
    '늘이다': ['늘리다'],
    '늘리다': ['늘이다'],
    '늘그막': ['늙으막'],
    '누다': ['놓다'],
    '너희': ['너네','니네','늬들','니들'],
    '니까': ['닌까'],
    '하니까': ['하닌까'],
    '다리다': ['달이다'],
    '달이다': ['다리다'],
    '다시피': ['다싶이'],
    '봬요': ['뵈요'],
    '뵈어요': ['뵈요'],
    '다행히': ['다행이'],
    '닦달하다': ['닥달하다'],
    '단출하다': ['단촐하다'],
    '닫히다': ['닫기다'],
    '달곰씁쓸하다': ['달콤씁쓸하다'],
    '달콤하다': ['달달하다'],
    '달착지근하다': ['달달하다'],
    '닭개장': ['닭계장'],
    '닭 볏': ['닭 벼슬'],
    '담그다': ['담구다'],
    '담갔다': ['담궜다'],
    '당최': ['당체','당췌'],
    '때문에': ['덕분에'],
    '덮밥': ['덥밥'],
    '덩굴': ['덩쿨'],
    '넝쿨': ['덩쿨'],
    '도긴개긴': ['도찐개찐'],
    '돕다': ['도우다'],
    '돌멩이': ['돌맹이'],
    '둥이': ['동이'],
    '대갚음': ['되갚음'],
    '도리어': ['되려'],
    '되레': ['되려'],
    '두껍다': ['두텁다'],
    '두텁다': ['두껍다'],
    '뒤지다': ['뒈지다'],
    '뒤처지다': ['뒤쳐지다'],
    '뒤쳐지다': ['뒤처지다'],
    '목덜미': ['뒷목'],
    '들어내다': ['드러내다'],
    '들이키다': ['들이켜다'],
    '들이켜다': ['들이키다'],
    '들치다': ['들추다'],
    '들추다': ['들치다'],
    '등쌀': ['등살'],
    '따 놓은 당상': ['따 논 당상'],
    '떼어 놓은 당상': ['따 논 당상'],
    '떼어 내다': ['때어 내다'],
    '떼다': ['때다'],
    '딱따구리': ['딱다구리'],
    '달리다': ['딸리다'],
    '떡볶이': ['떢뽁이','떡뽁이','떡볶기','떡복기'],
    '떴다': ['떳다'],
    '때': ['떼'],
    '떼': ['때'],
    '떼려야 뗄 수 없다': ['뗄래야 뗄 수 없다'],
    '눈에 띈다': ['눈에 뛴다'],
    '로': ['러'],
    '로서': ['로써'],
    '로써': ['로서'],
    '마늘': ['마눌'],
    '말소': ['마소'],
    '마소': ['말소'],
    '맞히다': ['맛추다'],
    '맞추다': ['맞히다'],
    '맡다': ['맏다'],
    '말발': ['말빨'],
    '이야': ['야'],
    '이지': ['지'],
    '맛보기': ['맛배기'],
    '만신창이': ['망신창이'],
    '맡기다': ['맞기다'],
    '메다': ['매다'],
    '매다': ['메다'],
    '멧돼지': ['맷돼지'],
    '멍에': ['굴레'],
    '굴레': ['멍에'],
    '맷돌': ['멧돌'],
    '무슨 요일': ['몇 요일'],
    '며칠': ['몇일'],
    '뭐': ['모'],
    '머': ['모'],
    '모둠': ['모듬'],
    '모임': ['모듬'],
    '메밀국수': ['모밀국수'],
    '모으다': ['모우다'],
    '몰아주다': ['모와주다'],
    '모자라다': ['모자르다'],
    '몸뚱어리': ['몸뚱아리'],
    '무르팍': ['무릎팍'],
    '무엇': ['무었'],
    '무': ['무우'],
    '무릅쓰다': ['무릎쓰다'],
    '뭉개다': ['뭉게다'],
    '미루나무': ['미류나무'],
    '미미하다': ['미비하다'],
    '미비하다': ['미미하다'],
    '미숫가루': ['미싯가루'],
    '미처': ['미쳐'],
    '미쳐': ['미처'],
    '및': ['밑'],
    '밑': ['및'],
    '밑동': ['밑둥'],
    '바라다': ['바래다'],
    '바래다': ['바라다'],
    '바뀌었다': ['바꼈다'],
    '바닥': ['바닦'],
    '박이다': ['박히다'],
    '박히다': ['박이다'],
    '바치다': ['받치다','받히다','밭치다'],
    '받히다': ['받치다','바치다','밭치다'],
    '밭치다': ['받치다','받히다','바치다'],
    '받치다': ['바치다','받히다','밭치다'],
    '밤새다': ['밤새우다'],
    '밤새우다': ['밤새다'],
    '밝히다': ['발키다'],
    '방정맞다': ['방정하다'],
    '방정하다': ['방정맞다'],
    '벙벙하다': ['벙찌다'],
    '봉숭아': ['봉숭화'],
    '봉선화': ['봉숭화'],
    '부딪히다': ['부딪치다'],
    '부딪치다': ['부딪히다'],
    '부시다': ['부수다'],
    '부수다': ['부시다'],
    '부풂': ['부품'],
    '불리다': ['불리우다'],
    '불거지다': ['붉어지다'],
    '붉어지다': ['불거지다'],
    '비린내': ['비릿내'],
    '비릿한 냄새': ['비릿내'],
    '비스름': ['비스무리'],
    '비껴가다': ['비켜가다'],
    '비켜가다': ['비껴가다'],
    '비위 상하다': ['빈정 상하다'],
    '빌리다': ['빌다'],
    '빌다': ['빌리다'],
    '빠트리다': ['빠치다'],
    '빠뜨리다': ['빠치다'],
    '빼닮다': ['빼박다'],
    '빼다박다': ['빼박다'],
    '뼈아프다': ['뼈 아프다'],
    '사달이 났다': ['사단이 났다'],
    '사글세': ['삭월세'],
    '사귀었다': ['사겼다'],
    '살코기': ['살고기'],
    '삼가다': ['삼가하다'],
    '삼촌': ['삼춘'],
    '섣부르다': ['섯부르다', '섲부르다'],
    '생토끼': ['새앙토끼'],
    '셍쥐': ['새앙쥐'],
    '새침데기': ['새침떼기'],
    '남색': ['곤색'],
    '빨강': ['빨강색'],
    '빨간색': ['빨강색'],
    '하늘색': ['소라색'],
    '섬뜩하다': ['섬짓하다'],
    '섬찟하다': ['섬짓하다'],
    '소꿉놀이': ['소꼽놀이'],
    '쇠다': ['쉬다','세다','새다'],
    '쉬다': ['쇠다'],
    '숟가락': ['숫가락'],
    '젓가락': ['젇가락'],
    '시오': ['시요'],
    '스라소니': ['시라소니'],
    '싸이다': ['쌓이다'],
    '쌓이다': ['싸이다'],
    '생뚱맞다': ['쌩뚱맞다'],
    '쓰이다': ['쓰이지다'],
    '쓰레기': ['쓰래기'],
    '쓸데없다': ['쓸 때 없다'],
    '씌다': ['씌이다'],
    '쓰잘머리': ['쓰잘데기'],
    '씻다': ['씼다'],
    '썩다': ['썪다'],
    '아니라고': ['아니다고'],
    '이라고': ['이다고'],
    '아는 척하다': ['알은척하다'],
    '아는 체하다': ['알은체하다'],
    '알은척하다': ['아는 척하다'],
    '알은체하다': ['아는 체하다'],
    '아등바등': ['아둥바둥'],
    '아예': ['아얘'],
    '악바리': ['악발이'],
    '아니': ['않이'],
    '압존법': ['앞존법'],
    '아기': ['애기'],
    '얘기': ['애기'],
    '예기': ['얘기'],
    '앳되다': ['애띠다'],
    '애당초': ['애시당초'],
    '아비': ['애비'],
    '얻다 대고': ['어따 대고'],
    '어리바리하다': ['어리버리하다'],
    '어쭙잖다': ['어줍잖다'],
    '얼마큼': ['얼만큼'],
    '얽히고설키다': ['얽히고 섥히다'],
    '엉키다': ['엉기다'],
    '엉기다': ['엉키다'],
    '어미': ['애미','에미'],
    '여우': ['여시'],
    '연거푸': ['연거퍼'],
    '염두에 두다': ['염두해 두다', '염두하다'],
    '요강': ['오강'],
    '오두방정': ['오도방정'],
    '오도독뼈': ['오돌뼈'],
    '오뚝이': ['오뚜기'],
    '오므리다': ['오무리다'],
    '아귀아귀': ['와구와구'],
    '웬만하다': ['왠만하다'],
    '왠지': ['웬지'],
    '외골수': ['외곬'],
    '외곬': ['외골수'],
    '반복해 외다': ['반복해 외우다'],
    '욕지기': ['욕지거리'],
    '우레': ['우뢰'],
    '우려먹다': ['울궈먹다'],
    '유래': ['유례'],
    '유례': ['유래'],
    '육개장': ['육계장'],
    '으레': ['으례'],
    '으스대다': ['으시대다'],
    '으스스하다': ['으시시하다'],
    '이부자리': ['이브자리'],
    '이른바': ['이름바'],
    '이래 봬도': ['이래뵈도'],
    '앎이': ['암이'],
    '이': ['이빨'],
    '머리': ['대가리'],
    '입': ['주둥이'],
    '이음매': ['이음새'],
    '이음새': ['이음매'],
    '일부로': ['일부러'],
    '일부러': ['일부로'],
    '함부로': ['함부러'],
    '일절': ['일체'],
    '일체': ['일절'],
    '일컬어': ['일컫어'],
    '일컫다': ['일컷다'],
    '입다': ['신다'],
    '신다': ['입다'],
    '잊다': ['잃다'],
    '잃다': ['잊다'],
    '아가씨': ['애기씨'],
    '자국': ['자욱'],
    '잠그다': ['잠구다'],
    '잠갔다': ['잠궜다'],
    '잠가': ['잠궈'],
    '제재': ['재제'],
    '재제': ['제재'],
    '제가': ['저가'],
    '적이': ['저으기'],
    '우리나라': ['저희 나라'],
    '전통': ['정통'],
    '정통': ['전통'],
    '제치다': ['제끼다'],
    '젖히다': ['제끼다'],
    '졸이다': ['조리다'],
    '조리다': ['졸이다'],
    '조몰락': ['조물락','조물딱'],
    '주쳇덩어리': ['주책덩어리'],
    '주워 먹다': ['줏어 먹다'],
    '주은': ['줏은'],
    '줍습니다': ['주웁니다'],
    '지루하다': ['지리하다'],
    '계집애': ['지지배'],
    '진실한': ['진실된'],
    '짓궂다': ['짓굳다','짓궃다','짖굳다','짖궂다','짖궃다'],
    '지르밟다': ['즈려밟다'],
    '자잘하다': ['짜잘하다'],
    '짜깁기': ['짜집기'],
    '잘리다': ['짤리다'],
    '쩨쩨하다': ['째째하다'],
    '주꾸미': ['쭈꾸미'],
    '찌개': ['찌게'],
    '지질하다': ['찌질하다'],
    '책거리': ['책걸이'],
    '쳐들어오다': ['처부수다'],
    '체': ['채'],
    '채': ['체'],
    '채신머리없다': ['체신머리없다'],
    '처먹다': ['쳐먹다'],
    '초주검': ['초죽음'],
    '예감': ['촉'],
    '총부리': ['총뿌리'],
    '치고받다': ['치고박다'],
    '치러': ['치뤄'],
    '칠칠하다': ['칠칠찮다','칠칠치 못하다'],
    '칠칠맞다': ['칠칠찮다', '칠칠치 못하다'],
    '칠칠찮다': ['칠칠하다','칠칠맞다'],
    '편': ['켠'],
    '쪽': ['켠'],
    '키읔': ['키역','키옄'],
    '티읕': ['티귿','티긑'],
    '헛물켜다': ['헛물키다'],
    '켜다': ['키다'],
    '먼지떨이': ['먼지털이'],
    '재떨이': ['재털이'],
    '털다': ['떨다'],
    '떨다': ['털다'],
    '텃새': ['텃세'],
    '텃세': ['텃새'],
    '통째로': ['통채로'],
    '통틀어서': ['통털어서'],
    '톡톡대다': ['틱틱대다'],
    '툭툭대다': ['틱틱대다'],
    '톡톡거리다': ['틱틱거리다'],
    '툭툭거리다': ['틱틱거리다'],
    '피다': ['피우다'],
    '피우다': ['피다'],
    '하나': ['허나'],
    '하네요': ['하내요'],
    '느라고': ['느냐고'],
    '느냐고': ['느라고'],
    '하양': ['하향'],
    '하향': ['하양'],
    '헌데': ['한데'],
    '한데': ['헌데'],
    '한술': ['한 수'],
    '한편': ['한켠'],
    '한구석': ['한켠'],
    '할퀴다': ['핡퀴다'],
    '항균': ['향균'],
    '핥다': ['햝다'],
    '해코지': ['해꼬지'],
    '핼쑥하다': ['핼쓱하다'],
    '햇빛': ['햇볕'],
    '햇볕': ['햇빛'],
    '허드레': ['허드래'],
    '해롱해롱': ['헤롱헤롱'],
    '혁대': ['혁띠'],
    '가죽띠': ['혁띠'],
    '호루라기': ['호르라기'],
    '화제': ['화재'],
    '화재': ['화제'],
    '훼손': ['회손'],
    '횡행': ['횡횡'],
    '휴면': ['휴먼'],
    '흉측하다': ['흉칙하다'],
    '흐리멍덩하다': ['흐리멍텅하다'],
    '희한하다': ['희안하다'],
    # 추가2
    '바람': ['바램'],
    '어이': ['어의'],
    '오랜만': ['오랫만'],
    '얘기': ['예기'],
    '웬일': ['왠일'],
    '며칠': ['몇일'],
    '교향곡': ['교양곡'],
    '훼손': ['회손'],
    '머리말': ['머릿말'],
    '끗발': ['끝발'],
    '대갚음': ['되갚음'], 
    '띄어쓰기': ['띄어 쓰기', '뛰어쓰기'],
    '붙여 쓰기': ['붙여쓰기'],
    '목말': ['목마'],
    '범칙': ['벌칙'],
    '악천후': ['악천우'],
    '결제': ['결재'],
    '결재': ['결제'],
    '지향': ['지양'],
    '지양': ['지향'],
    '연루': ['연류'],
    '늘그막': ['늙으막'],
    '어폐': ['어패'],
    '찌개': ['찌게'],
    '검정': ['검은색', '검정색'],
    '죔죔': ['잼잼'],
    '하루': ['1루'],
    '이틀': ['2틀'],
    '사흘': ['4흘'],
    '설렘': ['설램'],
    '한가락': ['한가닥'],
    '곁땀': ['겨땀'],
    '녘': ['녁'],
    '뒤치다꺼리': ['뒤치닥거리'],
    '빈털터리': ['빈털털이'],
    '설거지': ['설겆이', '설걷이'],
    '손톱깎이': ['손톱깎기'],
    '숨바꼭질': ['숨박꼭질'],
    '숯': ['숱'],
    '숱': ['숱'],
    '오뚝이': ['오뚜기'],
    '오지랖': ['오지랍'],
    '움큼': ['웅큼'],
    '웃어른': ['윗어른'],
    '윷놀이': ['윳놀이'],
    '벌거숭이': ['발가송이'],
    '개수': ['갯수'],
    '대가': ['댓가'],
    '대꾸': ['댓꾸'],
    '뒤태': ['뒷태'],
    '반대말': ['반댓말'],
    '시가': ['싯가'],
    '초점': ['촛점'],
    '해님': ['햇님'],
    '최솟값': ['최소값'],
    '최댓값': ['최대값'],
    '대게': ['대개', '되게'],
    '요새': ['요세'],
    '감안': ['가만'],
    '건더기': ['건데기'],
    '곱빼기': ['곱배기'],
    '구레나룻': ['구렛나루'],
    '낳다': ['낫다'],
    '낫다': ['낫다'],
    '가리키다': ['가르치다'],
    '되다': ['돼다'],
    '됐다': ['됬다'],
    '쩨쩨하다': ['몇일'],
    '깎다': ['교양곡'],
    '결딴나다': ['회손'],
    '틀리다': ['머릿말'],
    '헤매다': ['끝발'],
    '짜지다': ['짜여지다'], 
    '닿다': ['닫다'],
    '뺏다': ['뺏다'],
    '뺐다': ['뺏다'],
    '치르다': ['치루다'],
    '안절부절못하다': ['안절부절하다'],
    '서슴다': ['서슴하다'],
    '돋우다': ['돋구다'],
    '사사하다': ['사사받다'],
    '개다': ['개이다'],
    '되뇌다': ['되뇌이다'],
    '맞추다': ['마추다'],
    '붙이다': ['부치다'],
    '부치다': ['붙이다'],
    '싣다': ['실다'],
    '욱여넣다': ['우겨넣다'],
    '도와': ['도워'],
    '고와': ['고워'],
    '베풀다': ['배풀다'],
    '감질나다': ['감질맛나다'],
    '갇힌': ['갖힌'],
    '가진': ['갖은'],
    '건들다': ['건들이다'],
    '곯아떨어지다': ['골아떨어지다'],
    '구시렁거리다': ['궁시렁거리다'],
    '까무러치다': ['까무라치다'],
    '붇다': ['붓다'],
    '붓다': ['붇다'],
    '일부러': ['일부로'],
    '함부로': ['함부러'],
    '으레': ['으례'],
    '이따가': ['있다가'],
    '굽실': ['굽신'],
    '간간이': ['교양곡'],
    '간간히': ['간간이'],
    '간간이': ['간간히'],
    '곰곰이': ['곰곰히'],
    '번번이': ['번번히'],
    '일일이': ['일일히'], 
    '틈틈이': ['틈틈히'],
    '깨끗이': ['깨끗히', '깨끗하게'],
    '따뜻이': ['따뜻히', '따뜻하게'],
    '굳이': ['구지', '궂이'],
    '똑똑히': ['똑똑이'],
    '묵묵히': ['묵묵이'],
    '굽이굽이': ['구비구비'],
    '더욱이': ['더우기'],
    '반드시': ['반듯이'],
    '반듯이': ['반드시'],
    '아무튼': ['이뭏던', '아뭏든'],
    '일찍이': ['일찌기'],
    '지그시': ['지긋이'],
    '지긋이': ['지그시'],
    '깡충깡충': ['깡총깡총'],
    '오순도순': ['오손도손'],
    '오랜만': ['오랫만'],
    '대개': ['대게'],
     '싹둑싹둑': ['싹독싹독'],
    '가까워': ['가까와'],
    '되게': ['대개'],
    '괜스레': ['괜시리'],
    '그다지': ['그닥'],
    '그제야': ['그제서야'],
    '금세': ['금새'],
    '세다': ['새다', '쌔다'],
    '무난하다': ['문안하다'],
    '희한하다': ['희안하다'],
    '몹쓸다': ['목쓸다'],    
    '걸맞은': ['걸맞는'],
    '알맞은': ['알맞는'],
    '넉넉지': ['넉넉치'],
    '탐탁지': ['탐탁치'],
    '비뚤다': ['비뚫다'],
    '엔간하다': ['엥간하다'], 
    '졸리다': ['졸립다'],
    '거친': ['거칠은'],
    '괄시하다': ['괄세하다'],
    '괘씸하다': ['괴씸하다'],
    '률': ['율'],
    '율': ['률'],
    '안': ['않'],
    '않': ['안'],
    '애계': ['에게'],
    '에계': ['애걔']
}

# insulting_dict = ['씨발', '시발', 'ㅅㅂ']


# confusing_dict에 '다' 제거한 key와 value 생성, value의 첫 인덱스에 "품사" 추가
from copy import deepcopy
from konlpy.tag import Okt

okt = Okt()

confusing_dict_pos = {}
for confuse_typo in confusing_dict:
  change_lst = deepcopy(confusing_dict.get(confuse_typo))
  change_lst.insert(0,tuple(okt.pos(confuse_typo)))
  confusing_dict_pos[confuse_typo] = change_lst

  if confuse_typo[-1] == '다':
    change_lst2 = []
    confuse_typo2 = confuse_typo[:len(confuse_typo)-1] # confuse_typo 단어 마지막에 '~다' 뺀 상태
    change_lst2.append(tuple(okt.pos(confuse_typo2)))

    for i in range(1, len(change_lst)):
      change_word = change_lst[i]
      change_lst2.append(change_word[:len(change_word)-1]) # change_word 단어 마지막에 '~다' 뺀 상태를 append
    confusing_dict_pos[confuse_typo2] = change_lst2



kor_begin = 44032
kor_end = 55203
chosung_base = 588
jungsung_base = 28
jaum_begin = 12593
jaum_end = 12622
moum_begin = 12623
moum_end = 12643

chosung_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ',
                'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

jungsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ',
                 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ',
                 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ',
                 'ㅡ', 'ㅢ', 'ㅣ']

jongsung_list = [
    ' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ',
    'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ',
    'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ',
    'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

jaum_list = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄸ', 'ㄹ',
             'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ',
             'ㅃ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

moum_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
             'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

possible_char = [chosung_list, jungsung_list, jongsung_list]

def compose(chosung, jungsung, jongsung):
    char = chr(
        kor_begin +
        chosung_base * chosung_list.index(chosung) +
        jungsung_base * jungsung_list.index(jungsung) +
        jongsung_list.index(jongsung)
    )
    return char


def decompose(c): # ' ', '아', '안', 'ㄱ', 'ㅅ', '?', '.', '!' ... 등이 입력파라미터 c 예시
    if not character_is_korean(c): # ' '(공백), '.', '?', '!' 인 경우
        return (c)
    i = ord(c)
    if (jaum_begin <= i <= jaum_end): # 'ㄱ', 'ㅅ' ... 인 경우
        return (c)
    if (moum_begin <= i <= moum_end): # 'ㅗ', 'ㅙ' ... 인 경우
        return (c)

    #print(i) #####################################

    # decomposition rule => '안' 인 경우 예시
    i -= kor_begin
    cho = i // chosung_base # 'ㅇ'
    jung = (i - cho * chosung_base) // jungsung_base # 'ㅏ'
    jong = (i - cho * chosung_base - jung * jungsung_base) # 'ㄴ'
    return [chosung_list[cho], jungsung_list[jung], jongsung_list[jong]] # ['ㅇ','ㅏ','ㄴ']


def character_is_korean(c):
    i = ord(c)
    return ((kor_begin <= i <= kor_end) or
            (jaum_begin <= i <= jaum_end) or
            (moum_begin <= i <= moum_end))


def separate_jamo(sentence):
    ret = []
    for char in sentence:
        if char == "": continue
        ret.append(decompose(char))

    return ret


def merge_char(parsed_char):
    if len(parsed_char) == 3:
        return compose(*parsed_char)
    else:
        return parsed_char[0]


hangul_error_dict = dict()
## 자음
hangul_error_dict['ㅂ'] = ['ㅃ', 'ㅈ', 'ㅁ']
hangul_error_dict['ㅈ'] = ['ㅉ', 'ㅂ', 'ㄷ', 'ㄴ']
hangul_error_dict['ㄷ'] = ['ㄸ', 'ㅈ', 'ㄱ', 'ㅇ']
hangul_error_dict['ㄱ'] = ['ㄲ', 'ㄷ', 'ㅅ', 'ㄹ']
hangul_error_dict['ㅅ'] = ['ㅆ', 'ㄱ', 'ㅎ']
hangul_error_dict['ㅁ'] = ['ㅂ', 'ㄴ', 'ㅋ']
hangul_error_dict['ㄴ'] = ['ㅁ', 'ㅈ', 'ㅇ', 'ㅌ', 'ㅋ']
hangul_error_dict['ㅇ'] = ['ㄷ', 'ㄴ', 'ㅌ', 'ㅊ', 'ㄹ']
hangul_error_dict['ㄹ'] = ['ㅇ', 'ㄱ', 'ㅎ', 'ㅊ', 'ㅍ']
hangul_error_dict['ㅎ'] = ['ㄹ', 'ㅅ', 'ㅍ']
hangul_error_dict['ㅋ'] = ['ㅁ', 'ㅌ', 'ㄴ']
hangul_error_dict['ㅌ'] = ['ㅋ', 'ㅇ', 'ㅊ']
hangul_error_dict['ㅊ'] = ['ㅌ', 'ㅍ', 'ㄹ']
hangul_error_dict['ㅍ'] = ['ㅊ', 'ㄹ', 'ㅎ']

hangul_error_dict['ㅃ'] = ['ㅂ', 'ㅉ']
hangul_error_dict['ㅉ'] = ['ㅈ', 'ㅃ', 'ㄸ']
hangul_error_dict['ㄸ'] = ['ㄷ', 'ㅉ', 'ㄲ']
hangul_error_dict['ㄲ'] = ['ㄱ', 'ㄸ', 'ㅆ']
hangul_error_dict['ㅆ'] = ['ㅅ', 'ㄱ']

## 모음
hangul_error_dict['ㅛ'] = ['ㅗ', 'ㅕ']
hangul_error_dict['ㅕ'] = ['ㅛ', 'ㅗ', 'ㅓ', 'ㅑ']
hangul_error_dict['ㅑ'] = ['ㅕ', 'ㅏ', 'ㅐ']
hangul_error_dict['ㅐ'] = ['ㅒ', 'ㅑ', 'ㅔ', 'ㅣ']
hangul_error_dict['ㅔ'] = ['ㅖ', 'ㅐ', 'ㅢ']
hangul_error_dict['ㅗ'] = ['ㅛ', 'ㅓ', 'ㅜ']
hangul_error_dict['ㅓ'] = ['ㅕ', 'ㅏ']
hangul_error_dict['ㅏ'] = ['ㅑ', 'ㅣ', 'ㅓ']
hangul_error_dict['ㅣ'] = ['ㅐ']
hangul_error_dict['ㅠ'] = ['ㅜ', 'ㅗ', 'ㅡ']
hangul_error_dict['ㅜ'] = ['ㅠ', 'ㅡ']
hangul_error_dict['ㅡ'] = ['ㅜ']

hangul_error_dict['ㅒ'] = ['ㅐ', 'ㅖ', 'ㅔ']
hangul_error_dict['ㅖ'] = ['ㅔ', 'ㅐ', 'ㅒ']
hangul_error_dict['ㅘ'] = ['ㅚ']
hangul_error_dict['ㅙ'] = ['ㅞ']
hangul_error_dict['ㅚ'] = ['ㅙ', 'ㅘ', 'ㅟ']
hangul_error_dict['ㅝ'] = ['ㅞ']
hangul_error_dict['ㅞ'] = ['ㅙ']
hangul_error_dict['ㅟ'] = ['ㅚ', 'ㅢ']
hangul_error_dict['ㅢ'] = ['ㅟ', 'ㅔ']

## 종성 붙이는 경우
hangul_error_dict[' '] = ['ㅇ', 'ㄴ', 'ㅁ']


class NoiseInjector(object):

    def __init__(self, corpus, shuffle_sigma=0.01,
                 replace_mean=0.05, replace_std=0.03,
                 delete_mean=0.05, delete_std=0.03,
                 add_mean=0.05, add_std=0.03,
                 jamo_typo_mean=0.07, jamo_typo_std=0.03,
                 yuneum_typo_prob=0.07,
                 last_cute_prob=0.15,
                 punctuation_prob=0.1,
                 delete_space_prob=0.3,
                 add_space_prob=0.3,
                 confusing_prob=0.35):
        # READ-ONLY, do not modify
        self.corpus = corpus
        self.shuffle_sigma = shuffle_sigma
        self.replace_a, self.replace_b = self._solve_ab_given_mean_var(replace_mean, replace_std ** 2)
        self.delete_a, self.delete_b = self._solve_ab_given_mean_var(delete_mean, delete_std ** 2)
        self.add_a, self.add_b = self._solve_ab_given_mean_var(add_mean, add_std ** 2)
        self.jamo_typo_a, self.jamo_typo_b = self._solve_ab_given_mean_var(jamo_typo_mean, jamo_typo_std ** 2)
        self.yuneum_typo_prob = yuneum_typo_prob
        self.last_cute_prob = last_cute_prob
        self.punctuation_prob = punctuation_prob
        self.confusing_prob = confusing_prob
        self.delete_space_prob = delete_space_prob
        self.add_space_prob = add_space_prob
        # self.confusing_typo_a, self.confusing_typo_b = self._solve_ab_given_mean_var(confusing_typo_mean, confusing_typo_std**2)

    @staticmethod
    def _solve_ab_given_mean_var(mean, var):
        a = mean * mean * (1. - mean) / var - mean
        b = (1. - mean) * (mean * (1. - mean) / var - 1.)
        return a, b

    def _shuffle_func(self, tgt):
        if self.shuffle_sigma < 1e-6:
            return tgt

        shuffle_key = [i + np.random.normal(loc=0, scale=self.shuffle_sigma) for i in range(len(tgt))]
        new_idx = np.argsort(shuffle_key)
        res = [tgt[i] for i in new_idx]

        return res

    def _replace_func(self, tgt):
        replace_ratio = np.random.beta(self.replace_a, self.replace_b)
        ret = []
        rnd = np.random.random(len(tgt))
        for i, w in enumerate(tgt):
            if rnd[i] < replace_ratio:
                rnd_ex = self.corpus[np.random.randint(len(self.corpus))]
                rnd_word = rnd_ex[np.random.randint(len(rnd_ex))]
                ret.append(rnd_word)
            else:
                ret.append(w)
        return ret

    def _delete_func(self, tgt):
        delete_ratio = np.random.beta(self.delete_a, self.delete_b)
        ret = []
        rnd = np.random.random(len(tgt))
        for i, w in enumerate(tgt):
            if rnd[i] < delete_ratio:
                continue
            ret.append(w)
        return ret

    def _add_func(self, tgt):
        add_ratio = np.random.beta(self.add_a, self.add_b)
        ret = []
        rnd = np.random.random(len(tgt))
        for i, w in enumerate(tgt):
            if rnd[i] < add_ratio:
                rnd_ex = self.corpus[np.random.randint(len(self.corpus))]
                rnd_word = rnd_ex[np.random.randint(len(rnd_ex))]
                ret.append(rnd_word)
            ret.append(w)

        return ret

    def _delete_space_func(self, tgt):
        ret = []
        for i in range(len(tgt)):
            delete_ratio = random.uniform(0, 1)
            if tgt[i] == ' ' and delete_ratio < self.delete_space_prob:
                continue
            else:
                ret.append(tgt[i])

        return ret

    def _add_space_func(self, tgt):
        ret = []
        for i in range(len(tgt)):
            add_ratio = random.uniform(0, 1)
            if add_ratio < self.add_space_prob:
                ret.append(' ')
            ret.append(tgt[i])

        return ret

    def _jamo_typo_func(self, tgt): #tgt에는 [w for w in '안녕히'] 이런 형태가 들어감
        jamo_typo_ratio = np.random.beta(self.jamo_typo_a, self.jamo_typo_b)
        tgt = separate_jamo(tgt)
        ret = []
        rnd = np.random.random(len(tgt))
        for i, w in enumerate(tgt):
            decomposed_char = w
            if rnd[i] < jamo_typo_ratio and len(decomposed_char) == 3:
                if decomposed_char[2] == ' ':
                    typo_loc = np.random.choice(range(2))
                else:
                    typo_loc = np.random.choice(range(3))
                decomposed_char_modified = list(decomposed_char)

                if decomposed_char[typo_loc] in hangul_error_dict:
                    choice_error = np.random.choice(hangul_error_dict[decomposed_char[typo_loc]])
                    if choice_error in possible_char[typo_loc]:
                        decomposed_char_modified[typo_loc] = choice_error

                typo_word = merge_char(decomposed_char_modified)

                ret.append(typo_word)
            else:
                ret.append(merge_char(decomposed_char))

        return ret

    def _yuneum_typo_func(self, tgt):
        tgt = separate_jamo(tgt)

        ret = []
        ids = []
        for i in range(len(tgt) - 1):
            cur_char = tgt[i]
            next_char = tgt[i + 1]

            if len(next_char) == 3 and len(cur_char) == 3 and next_char[0] == 'ㅇ' and cur_char[2] != ' ' and cur_char[
                2] != 'ㅇ' \
                    and cur_char[2] in chosung_list:
                ids.append(i)

        selected_ids = random.sample(ids, int(len(ids) * self.yuneum_typo_prob))

        for i in selected_ids:
            cur_char = tgt[i]
            next_char = tgt[i + 1]

            tgt[i + 1][0] = cur_char[2]
            if tgt[i][2] != 'ㄴ':
                tgt[i][2] = ' '

        for i, p in enumerate(tgt):
            decomposed_char = p
            ret.append(merge_char(decomposed_char))

        return ret

    def _last_cute_func(self, tgt): #tgt에는 [w for w in '안녕히'] 이런 형태가 들어감
        cute_ratio = random.uniform(0, 1)
        if cute_ratio < self.last_cute_prob:
            last_loc = None
            for i in range(len(tgt) - 1, -1, -1):
                if character_is_korean(tgt[i]):
                    last_loc = i
                    break
            if last_loc != None:
                decomposed_char = decompose(tgt[last_loc])
                if len(decomposed_char) == 3 and decomposed_char[2] == ' ':
                    end_char = random.sample(['ㅁ', 'ㄴ', 'ㅇ', 'ㄹ', 'ㅅ', 'ㅋ', 'ㅎ'], 1)
                    decomposed_char[2] = end_char[0]
                    tgt[last_loc] = compose(*decomposed_char)

        return tgt

    def _punctuation_error_func(self, tgt):

        punc_list = ['.', '?', '!']

        for i in range(len(tgt)):
            if tgt[i] in punc_list:
                punc_ratio = random.uniform(0, 1)
                if punc_ratio < self.punctuation_prob:
                    tgt[i] = random.sample(punc_list + [''], 1)[0]

        return tgt

    def _confusing_error_func(self, word): #word에는 confusing_dict_pos.keys() 에 있는 글자('마음','세다','돼','되'... 등) 중 하나가 들어감

        confusing_ratio = random.uniform(0, 1)
        random_change_word = word

        changed = False
        if confusing_ratio < self.confusing_prob:
            changed = True
    
            # random.sample(confusing_dict_pos[word], 1)는 confusing_dict_pos[word] 값인 리스트에서 중복없이 뽑은 요소의 1 길이 리스트 반환
            # 예) random.sample(['모', '머'],1) 반환값은 ['모'] 혹은 ['머']
            # 참고 출처 : https://docs.python.org/ko/3/library/random.html
            random_change_word = random.sample(confusing_dict_pos[word][1:], 1)[0]

        return random_change_word, changed

    def _parse(self, pairs):
        align = []
        art = []
        for si in range(len(pairs)):
            ti = pairs[si][0]
            w = pairs[si][1]
            art.append(w)
            if ti >= 0:
                align.append('{}-{}'.format(si, ti))
        return art, align
    
    # 추가 코드
    def _is_valid_match(self, tokens_pos, tokens, confuse_start, confuse_end, confuse_typo):
          
          match_word_class_count = 0
          confuse_typo_classes = [ word_class for _,word_class in confusing_dict_pos[confuse_typo][0] ]
          #print(confuse_typo_classes)

          tokens_pos2 = tokens_pos.copy()
          tup_i = 0

          temp_i = 0
          start_i = 0
          end_i = 0
          for i,t in enumerate(tokens):
            if t == ' ':
              continue
            if t == tokens_pos2[tup_i][0][temp_i]:
              if temp_i == 0:
                start_i = i
              temp_i += 1

              if temp_i == len(tokens_pos2[tup_i][0]):
                end_i = i
                tokens_pos2[tup_i] = tokens_pos2[tup_i] + (start_i, end_i)

                tup_i += 1
                temp_i = 0

          #print(tokens_pos2)

          for morpheme, word_class, start_i, end_i in tokens_pos2:
            if confuse_start <= start_i < confuse_end:
              if word_class in confuse_typo_classes:
                match_word_class_count += 1
          
          ratio = 0.5
          #print(match_word_class_count)
          if ratio <= (match_word_class_count/len(confuse_typo_classes)):
            return True
          else:
            return False
      
    def inject_noise(self, tokens): # tokens에는 string이 들어간다 예) '안녕하세요'
        ##TODO: last cute 제일 마지막 character만 하도록
        # tgt is a vector of integers

        funcs = [self._add_func, self._shuffle_func, self._replace_func, self._delete_func]
        np.random.shuffle(funcs)
        funcs = [self._jamo_typo_func, self._yuneum_typo_func, self._punctuation_error_func] + funcs
        funcs_last = [self._jamo_typo_func, self._yuneum_typo_func, self._last_cute_func,
                      self._punctuation_error_func] + funcs

        token_parts = []
        token_change = []
        overlap = []
        # 추가 코드
        tokens_pos = okt.pos(tokens)
        for confuse_typo in confusing_dict_pos.keys():
            confuse_start = tokens.find(confuse_typo) # tokens는 find함수 사용하니까 => **string이구나!!**
            if confuse_start == -1: continue
            confuse_end = confuse_start + len(confuse_typo)
            # 추가 코드
            if not self._is_valid_match(tokens_pos, tokens, confuse_start, confuse_end, confuse_typo): continue

            check = False
            for over in overlap:
                if (over[0] <= confuse_start < over[1]) or (over[0] < confuse_end <= over[1]) or \
                        (confuse_start <= over[0] and over[1] < confuse_end):
                    check = True
            if check: continue
            overlap.append((confuse_start, confuse_end))

        overlap.sort()

        end = 0
        for ovlp in overlap:
            ovlp_st = ovlp[0]
            ovlp_ed = ovlp[1]
            # tokens[ovlp_st:ovlp_ed]는 반드시 confusing_dict_pos.keys() 에 있는 글자('마음','세다','돼','되'... 등) 중 하나
            # 왜냐면 앞에 for문에서 overlap.append((confuse_start, confuse_end)) 를 보면 이해할 수 있다!
            confusing_word = tokens[ovlp_st:ovlp_ed] 
            if end != ovlp_st:
                token_parts.append(tokens[end:ovlp_st])
                token_change.append(True)
            rndword, changed = self._confusing_error_func(confusing_word)
            token_parts.append(rndword)
            if changed:
                token_change.append(False)
            else:
                token_change.append(True)
            end = ovlp_ed
        token_parts.append(tokens[end:])
        token_change.append(True)

        # print("token_parts", token_parts) ###########################################3


        final_letters = []
        for idx, tokens in enumerate(token_parts):
            if token_change[idx] == False:
                final_letters.extend([w for w in tokens])
                continue
            letters = [w for w in tokens]

            # print("letters", letters) #####################################
            # print("funcs", funcs)  ############################################
            # print("func_last", funcs_last)  ############################################

            if idx != len(token_parts) - 1:
                for f in funcs:
                    letters = f(letters)
            else:
                for f in funcs_last:
                    letters = f(letters)
            final_letters.extend(letters)
        # print() ########################################
        return ''.join([letter for letter in final_letters])


def add_del_space(lines, add_ratio=0.1, del_ratio=0.3):
    noise_lines = []

    for line in lines:
        noise_line = []

        for token in line:
            if token == ' ':
                if random.uniform(0, 1) > del_ratio:  # not delete space
                    noise_line.append(' ')
            else:
                noise_line.append(token)
                if random.uniform(0, 1) < add_ratio:  # add space
                    noise_line.append(' ')

        noise_lines.append(''.join(noise_line))  # list -> string

    # print(f'[{lines[0]}] -> [{noise_lines[0]}]')

    return noise_lines

def noise(lines): # 수정 - 입력 파라미터
    tgts = lines
    noise_injector = NoiseInjector(tgts, shuffle_sigma=0, # 0.01
                 replace_mean=0.03, replace_std=0.01,  # 0.05 0.03
                 delete_mean=0.03, delete_std=0.01,    # 0.05 0.03
                 add_mean=0.03, add_std=0.01,          # 0.05 0.03
                 jamo_typo_mean=0.07, jamo_typo_std=0.03, # 0.03
                 yuneum_typo_prob=0.07, # 0.07
                 last_cute_prob=0.15, # 0.15
                 punctuation_prob=0.1, # 0.1
                 delete_space_prob=0.2, # 0.3
                 add_space_prob=0.2, # 0.3
                 confusing_prob=1) # 0.35
    
    # noise_injector2 = NoiseInjector(tgts, shuffle_sigma=0.001, # 0.01
    #              replace_mean=0.05, replace_std=0.03,  # 0.05 0.03
    #              delete_mean=0.05, delete_std=0.03,    # 0.05 0.03
    #              add_mean=0.05, add_std=0.03,          # 0.05 0.03
    #              jamo_typo_mean=0.1, jamo_typo_std=0.03, # 0.03
    #              yuneum_typo_prob=0.1, # 0.07
    #              last_cute_prob=0.15, # 0.15
    #              punctuation_prob=0.1, # 0.1
    #              delete_space_prob=0.2, # 0.3
    #              add_space_prob=0.3, # 0.3
    #              confusing_prob=0.4) # 0.35
    noise_lines = []

    

    end = 0
    cnt = 0
    
    for i in range(len(tgts)):
        tgt = tgts[i]
        noise_tgt = noise_injector.inject_noise(tgt)
        # noise_tgt2 = noise_injector2.inject_noise(tgt)
        prescriprive_tgt = g2p(tgt)
        noise_lines.append((noise_tgt,prescriprive_tgt))
   

    return noise_lines


nli_train_df = pd.read_csv(nli_train_csv_path)

from tqdm.notebook import tqdm
#from numba import jit, cuda


data_arr = np.empty((0,2),dtype='str')

for i in tqdm(range(400000,500000)):
    #print(nli_train_df.loc[i,'output'])
    sen = nli_train_df.loc[i,'output']
    tup = noise([sen])[0]
    data_arr = np.append(data_arr, np.array([[tup[0], sen]]), axis=0)
    data_arr = np.append(data_arr, np.array([[tup[1], sen]]), axis=0)
    if i % 1000 == 0:
        print(tup[0], sen)

nli_train_df2 = pd.DataFrame(data=data_arr, columns=['input','output'])
nli_train_df2.to_csv(nli_complete_train_csv_path_4, index=False)

print("="*100)

data_arr = np.empty((0,2), dtype='str')

for i in tqdm(range(500000,600000)):
    #print(nli_train_df.loc[i,'output'])
    sen = nli_train_df.loc[i,'output']
    tup = noise([sen])[0]
    data_arr = np.append(data_arr, np.array([[tup[0], sen]]), axis=0)
    data_arr = np.append(data_arr, np.array([[tup[1], sen]]), axis=0)
    if i % 1000 == 0:
        print(tup[0], sen)

nli_train_df2 = pd.DataFrame(data=data_arr, columns=['input','output'])
nli_train_df2.to_csv(nli_complete_train_csv_path_5, index=False)