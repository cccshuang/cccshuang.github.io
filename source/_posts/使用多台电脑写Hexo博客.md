---
title: 使用多台电脑写Hexo博客
date: 2018-09-28 14:36:53
categories:
- Hexo
tags: [Hexo]
---

最近觉得写Hexo博客都必须要打开自己的笔记本电脑很麻烦，想既可以在笔记本也可以在实验室的电脑上更新自己的博客，解决这个问题的主要方法是利用Git的分支，master分支用于发布和展示博客内容，并新建一个分支，这里起名比如叫"hexo",用于保存博客配置和博客markdown文件，供自己维护更新。
下面进行配置。

### 笔记本上配置方法
首先，我们已经在自己的笔记本上搭建好了Hexo博客，接下来登录Github，在yourusername.github.io仓库上新建一个分支,比如取名"hexo"，并切换到该分支，并在该仓库->Settings->Branches->Default branch中将默认分支设为"hexo"。
然后使用命令`git clone git@github.com:yourgithubname/yourname.github.io.git`将该仓库克隆到本地，在Git Bash中进入本地yourusername.github.io文件目录，执行`git branch`命令查看当前所在分支，应为新建的分支"hexo"。
将本地博客的部署文件（Hexo目录下的全部文件）全部拷贝进username.github.io文件目录中去，然后将themes目录以内中的主题的.git目录删除（如果有），因为一个git仓库中不能包含另一个git仓库，提交主题文件夹会失败，执行`git add .`、`git commit -m 'back up hexo files'`、`git push`即可将博客的hexo部署环境提交到GitHub个人仓库的xxx分支。
之后再在笔记本上写博客，即在username.github.io文件目录中进行了，这时需要在`npm install`一下。

### 实验室电脑配置方法

将新电脑的生成的ssh key添加到GitHub账户上
在新电脑上克隆yourname.github.io仓库的xxx分支到本地，此时本地git仓库处于xxx分支
切换到yourname.github.io目录，执行npm install(由于仓库有一个.gitignore文件，里面默认是忽略掉 node_modules文件夹的，也就是说仓库的hexo分支并没有存储该目录[也不需要]，所以需要install下)

这里，如果`npm install`出错，如"npm ERR! Unexpected end of JSON input while parsing near",可尝试：
- 删除package-lock.json文件
- 清除cache: `npm cache clean --force`
- 不要用淘宝镜像：`npm set registry https://registry.npmjs.org/`

### 发布更新博客
一切都没有问题后，我们在写完博客要发布的时候：
首先执行`git add .`、`git commit -m 'back up hexo files'`、`git push`指令，保证'hexo'分支版本最新。
执行`hexo d -g`指令，将博客更新到master分支。

注意：每次换电脑进行博客更新时，不管上次在其他电脑有没有更新，最好先`git pull`一下，按照上述步骤进行更新。
