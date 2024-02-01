# 节点记录

**2023.12.9--flask从0到实战--知了传课--已完21集--21中**

**2023.11.29--flask从0到实战--知了传课--已完11集**

**2023.11.29--flask从0到实战--知了传课--已完09集**

**2023.11.16--flask从0到实战--知了传课--已完06集**

**浏览器-->route-->view-->models(学习时可略过此步骤)-->view-->templates-->浏览器**

## 参数知识点  
    1. flask中都是关键字参数,<类型:参数名>
    2. 默认标识是尖括号<name>,同时name需要和视图函数中的参数一样
    3. 参数可以有默认值,如果有可以不用传递,没有默认值则必须要传递
    4. 默认参数是string类型
    5. 参数类型有:string,int,float,path,uuid,any;其中any表示元组中的任意一个做参数,<any('a','b','c'):an>
    6. methods请求方法:GET,POST,HEAD,PUT,DELETE
    7. url_for()--根据视图函数,反推url地址--url_for('函数名',参数名=value)

## request response

    前端发送一个请求request----后端服务器响应response
