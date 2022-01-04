from bs4 import BeautifulSoup
from selenium import webdriver 
from selenium.webdriver.common.keys import Keys
# filepath문제로 설치가 되어있어도 ipport에러가 뜨는 경우 발생. 파일경로를 수정해줘야한다.
# Ctrl + Shift + p 해서 창띄우고 select interpreter해서 현재 사용중인 가상환경 경로로 들어가서 파이썬 실행해준다.
import time 
import os 
import urllib.request

def createFolder(directory): 
    try: 
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)

#def choice(image):
    # 폴더를 띄워주고 사용할 사진의 번호를 입력받음 10장? 20장?
    
#1. 키워드 폴더 생성.
while True:
    
    keyword=input('키워드를 입력하세요 : ') 
    
    if keyword == 'exit':
        break;
    
    createFolder('./'+keyword+'_img_download') 
    chromedriver = 'C://chromedriver.exe' 
    driver = webdriver.Chrome(chromedriver) 
    driver.implicitly_wait(3)

    print(keyword, '검색') 
    driver.get('https://www.google.co.kr/imghp?hl=ko') 
    Keyword=driver.find_element_by_xpath('//*[@id="sbtc"]/div/div[2]/input') 
    Keyword.send_keys(keyword) 
    driver.find_element_by_xpath('//*[@id="sbtc"]/button').click()

    print(keyword+' 스크롤 중 .............') 
    elem = driver.find_element_by_tag_name("body") 
    for i in range(60): elem.send_keys(Keys.PAGE_DOWN) 
    time.sleep(0.1) 

    try: 
        driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div[1]/div[4]/div[2]/input').click() 
        for i in range(60): 
            elem.send_keys(Keys.PAGE_DOWN) 
            time.sleep(0.1) 
    except: 
        pass

    links=[] 
    images = driver.find_elements_by_css_selector("img.rg_i.Q4LuWd") 
    for image in images: 
        if image.get_attribute('src')!=None: 
            links.append(image.get_attribute('src')) 
            
    print(keyword+' 찾은 이미지 개수:',len(links)) 
    time.sleep(2)

    for k,i in enumerate(links): 
        url = i 
        start = time.time() 
        urllib.request.urlretrieve(url, "./"+keyword+"_img_download/"+keyword+"_"+str(k)+".jpg") 
        print(str(k+1)+'/'+str(len(links))+' '+keyword+' 다운로드 중....... Download time : '+str(time.time() - start)[:5]+' 초') 
    print(keyword+' ---다운로드 완료---') 
    driver.close()
    
    #.분류의 경계선 지정.
    num = int(input(f'{keyword}로 사용할 사진을 골라주세요 : '))
    
print('-------------키워드 폴더생성이 완료되었습니다.---------------')






    