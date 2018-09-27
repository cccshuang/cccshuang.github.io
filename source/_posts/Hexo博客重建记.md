---
title: Hexo博客重建记
date: 2018-08-04 23:29:38
categories:
- Hexo
tags: [Hexo]

---

### 前言
好记性是真的不如烂笔头！另外，自己也深刻觉得做事情多做10%十分重要，比如阅解决一个什么问题，不能懒，及时记录下来，下次遇到便能省不少事儿，而现实情况是每次遇到同一个问题还要重新折腾，很是难受；比如看书，及时整理记录下来，时而回顾，也能防止遗忘，节约不少功夫。
半年前兴趣突发用Hexo鼓捣了一个博客，结果之后就荒废了，再想用的时候发现什么都忘了，耽误很多时间，遂决定重建一下，并记录下来这个过程，一来以备自己随时查阅，二来可以给想建博客的人们一个参考。

### 准备工作
在安装 Hexo 之前，需要确保检查电脑中已安装下列软件：

- Node.js
- Git

有关Git和Node.js的安装可以参考廖雪峰的[Git教程](https://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000/00137396287703354d8c6c01c904c7d9ff056ae23da865a000)和[JavaScript教程](https://www.liaoxuefeng.com/wiki/001434446689867b27157e896e74d51a89c25cc8b43bdb3000/001434501245426ad4b91f2b880464ba876a8e3043fc8ef000)。对于windows用户来说，推荐使用 Git Bash 进行操作，这是git for windows自带的一组程序。

### 安装Hexo
接下来只需要运行下面一句命令即可完成 Hexo 的安装。
```
$ npm install -g hexo-cli
```

安装 Hexo 完成后，新建一个网站，执行：
```
$ hexo init Hexo_Blog  # Hexo_Blog是博客所在文件夹名，可自行替换为你的文件夹
$ cd Hexo_Blog
$ npm install
```

### 安装主题

设置主题，这里我采用的是非常受欢迎的[Next主题](http://theme-next.iissnan.com/)，运行下面命令，直接克隆整个仓库。
```
$ cd Hexo_Blog
$ git clone https://github.com/iissnan/hexo-theme-next themes/next
```

然后在**站点配置文件**中设置你的主题：
```
theme: next
```

接下来我们验证主题是否安装成功。
```
$ hexo clean
$ hexo s --debug
```
此时使用浏览器访问 http://localhost:4000，检查是否成功。如果端口4000被占用，可使用 -p 更换端口。

>在 Hexo 中有两份主要的配置文件，其名称都是 _config.yml。 其中，一份位于站点根目录下，主要包含 Hexo 本身的配置；另一份位于主题目录下，这份配置由主题作者提供，主要用于配置主题相关的选项。为了描述方便，在以下说明中，将前者称为 站点配置文件， 后者称为 主题配置文件。


### 配置主题
#### 选择 Scheme
Scheme 是 NexT 提供的一种特性，不同Scheme有不同的外观。切换只需在**主题配置文件**中更改即可。
```
# Schemes
scheme: Muse
#scheme: Mist
#scheme: Pisces
#scheme: Gemini
```
#### 设置站点名，语言，作者昵称，站点描述
在**站点配置文件**进行更改，如下：
```
# Site
title: shuang's blog
subtitle:
description: 滴水穿石，非一日之功
keywords:
author: cccshuang
language: zh-Hans
timezone:
```

#### 设置头像
将头像放置主题目录下的 source/images/ 目录下，在**主题配置文件**中配置为：
```
avatar: /images/avatar.gif
```

#### 设置菜单
在**主题配置文件**中修改以下内容：
```
menu:
  home: / || home
  #about: /about/ || user
  tags: /tags/ || tags
  categories: /categories/ || th
  archives: /archives/ || archive
```
其中分类、标签云、关于等页面需要自己添加，输入
```
hexo new page "categories" #新建页面
```
之后在站点目录下的source文件夹下，会新增一个categories的文件夹，里面有一个index.md文件，编辑如下：
```
title: 分类
date: 2018-08-04 15:44:40
type: "categories"
comments: false
```
comments设置为false是打开分类页面，不显示评论插件。
tags, about页面的创建相似，如出现中文乱码，可尝试把创建的md文件更改为UTF-8编码。

#### 设置首页列表是否显示阅读更多
在**主题配置文件**中修改以下内容：
```
auto_excerpt:
  enable: true
  length: 150
```

#### 添加RSS
首先安装 hexo-generator-feed插件：
```
$ npm install --save hexo-generator-feed
```
修改站点配置文件:
```
feed: # RSS
  type: atom
  path: atom.xml
  limit: 0

plugins: hexo-generate-feed
```
修改主题配置文件:
```
rss: /atom.xml
```
#### 添加侧边栏社交链接
修改主题配置文件:
```
social:
  GitHub: your github url
  ZhiHu: your zhihu url
```
#### 字数统计
统计文章的字数以及大致分析出阅读时间。修改主题配置文件：
```
post_wordcount:
  item_text: true
  wordcount: true
  min2read: true
  totalcount: true
  separated_meta: ture
```
并安装插件：
```
$ npm install hexo-wordcount --save
```

#### 背景动画
将主题配置文件下面其中一项改为true即可。
```
# Canvas-nest
canvas_nest: true
# three_waves
three_waves: false
# canvas_lines
canvas_lines: false
# canvas_sphere
canvas_sphere: false
```


#### 添加自动打开编辑器脚本
在 博客根目录/scripts/ 下新建 AutoOpenEditor.js 文件（取其他名字也可以，不影响）（如果没有 scripts 目录则新建），并粘贴以下代码，保存。
```
let spawn = require('hexo-util/lib/spawn');

hexo.on('new', (data) => {
  spawn('code', [hexo.base_dir, data.path]);
});
```
这样，在你每次 hexo new 的时候，脚本就会自动帮你打开 VS Code 并切换到博客根目录顺带打开新建的 .md 文件啦。
参考博客 [HEXO小技巧在 hexo new 的时候自动用 VS Code 打开新建文章](https://leaferx.online/2018/03/17/hexo-auto-open-vscode/)

### 第三方服务
#### 阅读次数统计（LeanCloud）
可参考这个博客 [为NexT主题添加文章阅读量统计功能](https://notes.wanghao.work/2015-10-21-%E4%B8%BANexT%E4%B8%BB%E9%A2%98%E6%B7%BB%E5%8A%A0%E6%96%87%E7%AB%A0%E9%98%85%E8%AF%BB%E9%87%8F%E7%BB%9F%E8%AE%A1%E5%8A%9F%E8%83%BD.html#%E9%85%8D%E7%BD%AELeanCloud)

#### 分享文章功能
使用[AddThis](https://www.addthis.com)，定义自己的样式，如可以通过微信，微博，qq等进行分享。然后在Profile Settings的General里复制ID，修改主题配置文件：
```
add_this_id: {your AddThis ID}
```

更多设置，可以参考博客[hexo搭建个人博客--NexT主题优化](https://www.jianshu.com/p/1f8107a8778c)


#### 数学公式
修改主题配置文件：
```
# MathJax Support
mathjax:
  enable: true
  per_page: true
  cdn: //cdn.bootcss.com/mathjax/2.7.1/latest.js?config=TeX-AMS-MML_HTMLorMML
```
在写博客时，如果博文带有公式，头部增加一项mathjax，如：
```
---
title: math
date: 2018-08-04 23:12:07
tags:
mathjax: true
---
```

### 部署到github
登录github，创建一个repo，名称为 “yourname.github.io”, 其中yourname是你的github名称。
在**站点配置文件**进行更改，如下：
```
deploy:
  type: git
  repo: git@github.com:yourname/yourname.github.io.git
  branch: master
```
然后安装一个插件
```
$ npm install hexo-deployer-git --save
```
执行命令:
```
$ hexo clean
$ hexo g
$ hexo d
```
或
```
$ hexo clean
$ hexo d -g #生成并上传
```
即可将你写好的文章部署到github服务器上，打开浏览器，输入http://yourgithubname.github.io 检测是否成功。

如果出现如下类似的错误：
```
error: failed to execute prompt script (exit code 1)
fatal: could not read Username for 'https://github.com': Invalid argument

    at ChildProcess.<anonymous> (H:\Hexo_Blog\node_modules\hexo-util\lib\spawn.js:37:17)
    at emitTwo (events.js:126:13)
    at ChildProcess.emit (events.js:214:7)
    at ChildProcess.cp.emit (H:\Hexo_Blog\node_modules\cross-spawn\lib\enoent.js:40:29)
    at maybeClose (internal/child_process.js:925:16)
    at Process.ChildProcess._handle.onexit (internal/child_process.js:209:5)

```
可以尝试重新[配置github账户信息](https://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000/00137396287703354d8c6c01c904c7d9ff056ae23da865a000)和[配置SSH](https://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000/001374385852170d9c7adf13c30429b9660d0eb689dd43a000)。


### 写博客
hexo根目录，执行命令：
```
$ hexo new 'first blog'
```
hexo会在\source\_posts下生成相关md文件，打开便可开始写博客了。博客格式如下：
```
---
title: MongoDB学习笔记
date: 2018-01-17 19:40:37
categories:
- Database
tags:
- MongoDB
- NoSQL
---
正文blabla
```

### 其他
卸载hexo
`$ npm uninstall hexo-cli -g`