#  Devops Git概述



## Devops

​	开发  development       运维     operations 

- Devops能干什么：提升产品质量

- 1. 自动化测试
  2. 集成测试
  3. 代码质量管理工具
  4. 程序员鼓励师

- Devops如何实现

  设计架构规划-代码的存储-构建-测试、预生产、部署、监控

  

## Git版本控制系统

- 简介

- 1. 版本控制系统是一种记录一个或若干个文件内容变化,以便将来查阅特定版本内容情况的系统 

  2. 记录文件的所有历史变化      

  3. 随时可恢复到任何一个历史状态

  4. 多人协作开发

     

- 常见的版本管理工具

- 1. SVN

     集中式的版本控制系统,只有一个中央数据仓库,如果中央数据仓库挂了或者不可访问,所有的使用者无法使用SVN,无法进行提交或备份文件。

     <img src="/Users/jin/Library/Application Support/typora-user-images/image-20210420234032436.png" alt="image-20210420234032436" style="zoom: 33%;" />

  2. Git 

     分布式的版本控制系统,在每个使用者电脑上就有一个完整的数据仓库,没有网络依然可以使用Git。当然为了习惯及团队协作,会将本地数据同步到Git服务器或者 Github等代码仓库。

     <img src="/Users/jin/Library/Application Support/typora-user-images/image-20210420234105026.png" alt="image-20210420234105026" style="zoom:33%;" />



##Git安装部署

### 系统环境准备

- CentOS

<img src="/Users/jin/Library/Application Support/typora-user-images/image-20210420234401642.png" alt="image-20210420234401642" style="zoom: 50%;" />

- MacOS

  ```bash
  (base) MacBook-Pro:~ jin$ git --version
  git version 2.21.1 (Apple
  ```

### GIt安装部署

1. ![image-20210420234656411](/Users/jin/Library/Application Support/typora-user-images/image-20210420234656411.png)

2. 创建一个全局用户名、全局邮箱作为配置信息

   ```bash
   # MacOS
   git config --global user.name 'sora01'    #配置git使用用户 
   git config --global user.email 'sora01@qq.com'   #配置git使用邮箱
   git config --global color.ui true   #语法高亮
   ```

3. Git初始化

   初始化工作目录、对已存在的目录或者对已存在的目录都可进行初始化

   1. 初始化

      ```bash
      [sora @ sora-localhost ~]  mkdir data
      [sora @ sora-localhost ~]  cd data/
      [sora @ sora-localhost ~/data]  git init      #初始化
      Initialized empty Git repository in /home/sora/data/.git/
      
      [sora @ sora-localhost ~/data]  ll -a
      total 4
      drwxrwxr-x   3 sora sora   18 Apr 21 23:43 .
      drwx------. 17 sora sora 4096 Apr 21 23:42 ..
      drwxrwxr-x   7 sora sora  119 Apr 21 23:43 .git
      ```

   2. 查看工作区状态

      ```bash
      git status		#查看工作区状态
      
      [sora @ sora-localhost ~/data]  tree .git/
      .git/
      ├── branches			#分支目录
      ├── config				#定义项目特有的配置选项
      ├── description		#仅供 git web程序使用
      ├── HEAD					#指示当前的分支
      ├── hooks					#包含git钩子文件
      │   ├── applypatch-msg.sample
      │   ├── commit-msg.sample
      │   ├── post-update.sample
      │   ├── pre-applypatch.sample
      │   ├── pre-commit.sample
      │   ├── prepare-commit-msg.sample
      │   ├── pre-push.sample
      │   ├── pre-rebase.sample
      │   └── update.sample
      ├── info					#包含一个全局排除文件( exclude文件)
      │   └── exclude
      ├── objects				#存放所有数据内容
      │   ├── info
      │   └── pack
      └── refs					#存放指向数据(分支)的提交对象的指针
          ├── heads
          └── tags
      #└── index 保存暂存区信息,在执行 git initl的时候,这个文件还没有
      ```

      

# Git常规使用

##创建数据-提交数据

![image-20210422101548471](/Users/jin/Library/Application Support/typora-user-images/image-20210422101548471.png)

## GIt基础命令

### 添加文件

````bash
[root @ sora-localhost ~/data]  touch a b c
[root @ sora-localhost ~/data]  git status
# Untracked files
#       a
#       b
#       c
[root @ sora-localhost ~/data]  git add a			   #添加文件到暂存区
# Changes to be committed:
#       new file:   a
#
# Untracked files:
#       b
#       c
[root @ sora-localhost ~/data]  git add .     	#添加当前所有文件到暂存区 
# Changes to be committed:
#
#       new file:   a
#       new file:   b
#       new file:   c

````
### 删除文件
```bash
[root @ sora-localhost ~/data]  git rm --cached c 		#撤出暂存区
rm 'c'
# Changes to be committed:
#       new file:   a
#       new file:   b
#
# Untracked files:
#
#       c
[root @ sora-localhost ~/data]  git rm -f b			#同时删除暂存区和工作目录的文件
```
### 提交文件
```bash
[root @ sora-localhost ~/data]  git commit -m "add newfile a"			#从暂存区提交到本地仓库
      [master (root-commit) 5c35cc1] add newfile a
       1 file changed, 0 insertions(+), 0 deletions(-)
       create mode 100644 a
[root @ sora-localhost ~/data]  git status 
      # On branch master
      nothing to commit, working directory clean
```

- 小结:如何真正意义上通过版本控制系统管理文件
  1. 工作目录必须有个代码文件
  
  2. 通过`git add <fi1e>`添加到暂存区域
  
  3. 通过 `git commit  -m  "你自己输入的信息"`添加到本地仓库
  
     

###修改文件名称

```bash
[root @ sora-localhost ~/data]  git mv a.txt a			#git命令直接更改名称
# Changes to be committed:
#       renamed:    a.txt -> a
[root @ sora-localhost ~/data]  git commit -m "rename a.txt a"
```

### 文件比对

```bash
[root @ sora-localhost ~/data]  echo 'sora你好!' >a
[root @ sora-localhost ~/data]  git diff 		#比对工作目录和暂存区有什么不同
    diff --git a/a b/a
    index e69de29..90d8bba 100644
    --- a/a
    +++ b/a
    @@ -0,0 +1 @@
    +sora你好!
    
    
[root @ sora-localhost ~/data]  git add a		
[root @ sora-localhost ~/data]  git diff 		#由于添加到暂存区 因此工作目录和暂存区相同 无返回
[root @ sora-localhost ~/data]  git diff --cached			#比对暂存区和本地仓库有什么不同
    diff --git a/a b/a
    index e69de29..90d8bba 100644
    --- a/a
    +++ b/a
    @@ -0,0 +1 @@
    +sora你好!
    
    
[root @ sora-localhost ~/data]  git commit -m "add greet"
    [master 364af5a] add greet
     1 file changed, 1 insertion(+)
[root @ sora-localhost ~/data]  git diff
[root @ sora-localhost ~/data]  git diff --cached		#由于提交到本地仓库 因此本地仓库和暂存区相同 
```

```bash
[root @ sora-localhost ~/data]  echo 123456 >> a 
[root @ sora-localhost ~/data]  git commit -am "add greet2"  
#如果某个文件已经被仓库管理,如果在更改此文件直接需要一条命令提交即可
    [master 4245e96] add greet2
     1 file changed, 1 insertion(+)
[root @ sora-localhost ~/data]  git diff --cached
```

### 查看历史快照

```bash
[root @ sora-localhost ~/data]  git log
      commit 4245e962027f9db38f4bc7dfe36f5f7089af228b   # 哈希值
      Author: sora02 <sora02@qq.com>										# 作者信息
      Date:   Thu Apr 22 12:05:35 2021 +0800						# 时间
          add greet2																		# 描述信息

[root @ sora-localhost ~/data]  git log --oneline				# 一行简单的显示 Commit信息
      4245e96 add greet2	
      364af5a add greet

[root @ sora-localhost ~/data]  git log --oneline --decorate			# 显示当前的指针指向哪里
      4245e96 (HEAD, master) add greet2
      364af5a add greet
      
[root @ sora-localhost ~/data]  git log  -p							# 显示具体内容的变化
[root @ sora-localhost ~/data]  git log  -1							# 只显示1条内容
```



### 恢复历史数据

```bash
[root @ sora-localhost ~/data]  git reset --hard 364af5a			 # 回滚数据到某一个提交
			HEAD is now at 364af5a add greet

[root @ sora-localhost ~/data]  git reflog 							# 可以查看所有分支的所有操作记录
      364af5a HEAD@{0}: reset: moving to 364af5a
      4245e96 HEAD@{1}: commit: add greet2
      364af5a HEAD@{2}: commit: add greet
      35ecfd5 HEAD@{3}: commit: rename a.txt a
      c1cdef6 HEAD@{4}: commit: rename a a.txt
      5c35cc1 HEAD@{5}: commit (initial): add newfile a
```



## Git分支

​		**分支**即是平行空间,假设你在为某个手机系统研发拍照功能,代码已经完成了80%,但如果将这不完整的代码直接提交到git仓库中,又有可能影响到其他人的工作,此时我们便可以在该软件的项目之上创建一个名叫”拍照功能”的分支,这种分支只会属于你自己,而其他人看不到,等代码编写完成后再与原来的项目主分支合并下即可,这样即能保证代码不丢失,又不影响其他人的工作。

![image-20210422135553437](/Users/jin/Library/Application Support/typora-user-images/image-20210422135553437.png)

​		一般在实际的项目开发中,我们要尽量保证 masters分支是非常稳定的,仅用于发布新版本,平时不要随便直接修改里面的数据文件。

​		而工作的时候则可以新建不同的工作分支,等到工作完成后在**合并**到 master分支上面,所以团队的合作分支看起来会像上面图那样。

### 创建 查看

```bash
[root @ sora-localhost ~/data]  git log --oneline --decorate	# 显示当前的指针指向哪里
      4245e96 (HEAD, master) add greet2
      364af5a add greet

[root @ sora-localhost ~/data]  git branch testing								# 创建分支
[root @ sora-localhost ~/data]  git branch												# 查看分支
      * master
        testing
```



### 切换分支

```bash
[root @ sora-localhost ~/data]  git checkout testing 							# 切换分支
      Switched to branch 'testing'
[root @ sora-localhost ~/data]  git branch 
        master
      * testing
```

### 删除分支

```bash
[root @ sora-localhost ~/data]  git branch -d testing  							# 删除分支
      Deleted branch testing (was 4245e96).
[root @ sora-localhost ~/data]  git branch 
			* master
```



### 操纵分支

![image-20210422151330684](/Users/jin/Library/Application Support/typora-user-images/image-20210422151330684.png)

```bash
## Step1.
[root @ sora-localhost ~/data]  touch aaa ...bbb...ccc
# 省略git add, git commit操作

## Step2.
[root @ sora-localhost ~/data]  git checkout -b testing						# 创建并切换到分支
			Switched to a new branch 'testing'
[root @ sora-localhost ~/data]  git branch 
        master
      * testing
      
[root @ sora-localhost ~/data]  git log --oneline --decorate
			44e28b5 (HEAD, testing, master) add ccc
			
## Step3.
[root @ sora-localhost ~/data]  touch testing-add
[root @ sora-localhost ~/data]  git add .
[root @ sora-localhost ~/data]  git commit -m 'add newfile testing-add'
[root @ sora-localhost ~/data]  git log --oneline --decorate
      3b3844f (HEAD, testing) add newfile testing-add
      44e28b5 (master) add ccc

## Step4.
[root @ sora-localhost ~/data]  git checkout master 
[root @ sora-localhost ~/data]  touch master-eee
[root @ sora-localhost ~/data]  git add .
[root @ sora-localhost ~/data]  git commit  -m "add newfile master-eee"
[root @ sora-localhost ~/data]  git log --oneline --decorate
			c8dc512 (HEAD, master) add newfile master-eee
```

![image-20210422152424990](/Users/jin/Library/Application Support/typora-user-images/image-20210422152424990.png)



### 合并分支

```bash
## Step5 6.
[root @ sora-localhost ~/data]  git merge testing						#合并分支
      Merge made by the 'recursive' strategy.
       testing-add | 0
       1 file changed, 0 insertions(+), 0 deletions(-)
       create mode 100644 testing-add
[root @ sora-localhost ~/data]  ll
      total 4
      -rw-r--r-- 1 root root  0 Apr 22 14:56 master-eee
      -rw-r--r-- 1 root root  0 Apr 22 14:59 testing-add
      
[root @ sora-localhost ~/data]  git branch -d testing 			# 合并完成后删除分支即可
			Deleted branch testing (was 3b3844f).
```

![image-20210422152518833](/Users/jin/Library/Application Support/typora-user-images/image-20210422152518833.png)



### 冲突合并

```bash
[root @ sora-localhost ~/data]  git checkout master 
			Switched to branch 'master'
[root @ sora-localhost ~/data]  echo master >> aaa
[root @ sora-localhost ~/data]  git commit -am "modify aaa add master"


[root @ sora-localhost ~/data]  git checkout testing 
			Switched to branch 'testing'
[root @ sora-localhost ~/data]  echo testing >> aaa
[root @ sora-localhost ~/data]  git commit -am "modify aaa add testing"


[root @ sora-localhost ~/data]  git checkout master 
[root @ sora-localhost ~/data]  git merge testing 					# 合并分支 aaa文件的内容发生了冲突
      Auto-merging aaa
      CONFLICT (content): Merge conflict in aaa
      Automatic merge failed; fix conflicts and then commit the result.
[root @ sora-localhost ~/data]  cat aaa
      <<<<<<< HEAD
      master
      =======
      testing
      >>>>>>> testing



[root @ sora-localhost ~/data]  vim aaa     # 冲突的文件自动标识到文件里,手动更改沖突要保留的代码
[root @ sora-localhost ~/data]  cat aaa
			master
[root @ sora-localhost ~/data]  git commit -am "resolve marge conflict"
			[master 99f8d5c] resolve marge conflict
```

![image-20210422170141404](/Users/jin/Library/Application Support/typora-user-images/image-20210422170141404.png)





## Git标签

​		标签也是指向了一次 `commit`提交,是一个里程碑式的标签,回滚打标签直接加标签号,不需要加唯一字符串不好记

```bash
[root @ sora-localhost ~/data]  git log --oneline --decorate
      99f8d5c (HEAD, master) resolve marge conflict
      ....
      4a8aa3a add aaa
[root @ sora-localhost ~/data]  git tag -a v1.0 4a8aa3a -m "Tag_add aaa"   
			# -a指定标签名   哈希值 指定某一次的提交为标签    -m指定说明文字
[root @ sora-localhost ~/data]  git show v1.0 			# 查看v1.0的信息 


[root @ sora-localhost ~/data]  git reset --hard v1.0     # 回滚数据到v1.0的提交
[root @ sora-localhost ~/data]  ll
      total 4
      -rw-r--r-- 1 root root 19 Apr 22 17:16 a
      -rw-r--r-- 1 root root  0 Apr 22 17:18 aaa


[root @ sora-localhost ~/data]  git reflog 
      4a8aa3a HEAD@{0}: reset: moving to v1.0
      99f8d5c HEAD@{1}: reset: moving to v2.0
      364af5a HEAD@{2}: reset: moving to v1.0
      99f8d5c HEAD@{3}: commit (merge): resolve marge conflict
[root @ sora-localhost ~/data]  git tag -a v2.0 99f8d5c -m "Tag_resolve marge conflict"

[root @ sora-localhost ~/data]  git reset --hard v2.0 
[root @ sora-localhost ~/data]  ll
			total 8
			
[root @ sora-localhost ~/data]  git tag    # 显示全部标签
      v1.0
      v2.0
[root @ sora-localhost ~/data]  git tag -d v1.0			# 删除标签
			Deleted tag 'v1.0' (was 5a77da4)
```







# GItHub

​		**Github**顾名思义是一个Git版本库的托管服务,是目前全球最大的软件仓库,拥有上百万的开发者用户,也是软件 开发和寻找资源的最佳途径, Github不仅可以托管各种Git版本仓库,还拥有了更美观的Mleb界面,您的代码文件可 以被任何人克隆,使得开发者为开源项贡献代码变得更加容易,当然也可以付费购买私有库,这样高性价比的私有 库真的是帮助到了很多团队和企业 

1. 注册用户 课前注册好用户 

2. 配置ssh-key 

   ````bash
   ssh-keygen -t rsa
   # 无限回车 生成公钥
   cat .ssh/id_rsa.pub
   # 获取ssh-key 将其复制 
   ````

   复制到GitHub设置里的SSH keys

   ![image-20210422231359118](/Users/jin/Library/Application Support/typora-user-images/image-20210422231359118.png)

3. 创建项目 

   ​	创建一个新的仓库 Create a new repository 步骤省略

   

4. 推送新代码到 github![image-20210422230738585](/Users/jin/Library/Application Support/typora-user-images/image-20210422230738585.png)

   ```bash
   [root @ sora-localhost ~/data]  git remote add origin git@github.com:Apotosome/git_data.git
   [root @ sora-localhost ~/data]  git push -u origin master
       ...
       Compressing objects: 100% (22/22), done.
       Writing objects: 100% (31/31), 2.51 KiB | 0 bytes/s, done.
       Total 31 (delta 8), reused 0 (delta 0)
       remote: Resolving deltas: 100% (8/8), done.
       To git@github.com:Apotosome/git_data.git
        * [new branch]      master -> master
       Branch master set up to track remote branch master from origin.
   ```

   ![image-20210423092654373](/Users/jin/Library/Application Support/typora-user-images/image-20210423092654373.png)

   

5. 克隆项目到本地 

<img src="/Users/jin/Library/Application Support/typora-user-images/image-20210423092936486.png" alt="image-20210423092936486" style="zoom:50%;" />

```bash
[root @ sora-localhost ~/dataTemp]  git clone git@github.com:Apotosome/git_data.git
      Cloning into 'git_data'...
      remote: Enumerating objects: 31, done.
      remote: Counting objects: 100% (31/31), done.
      remote: Compressing objects: 100% (14/14), done.
      remote: Total 31 (delta 8), reused 31 (delta 8), pack-reused 0
      Receiving objects: 100% (31/31), done.
      Resolving deltas: 100% (8/8), done.
[root @ sora-localhost ~/dataTemp]  ll git_data/
      total 8
      -rw-r--r-- 1 root root 19 Apr 23 09:32 a
      -rw-r--r-- 1 root root  7 Apr 23 09:32 aaa
      -rw-r--r-- 1 root root  0 Apr 23 09:32 bbb
      -rw-r--r-- 1 root root  0 Apr 23 09:32 ccc
      -rw-r--r-- 1 root root  0 Apr 23 09:32 master-eee
      -rw-r--r-- 1 root root  0 Apr 23 09:32 testing-add
```



# Gitlab

​		Gitlab是一个用于仓库管理系统的开源项目。使用Git作为代码管理工具,并在此基础上搭建起来的eb服务。可通过eb界面进行访问公开的或者私人项目。它拥有与 Github类似的功能,能够浏览源代码,管理缺陷和注释。可 以管理团队对仓库的访词,它非常易于浏览提交过的版本并提供一个文件历史库。团队成员可以利用内置的简单聊天程序进行交流。它还提供一个代码片段收集功能可以轻松实现代码复用。



## 安装Gitlab

1. 下载链接  选择对应的系统类型

   ```bash
   https://about.gitlab.com/install/
   ```

2. 安装和配置必要的依赖项;禁用防火墙,关 selinux

   ```bash
   sudo yum install -y curl policycoreutils-python openssh-server perl
   sudo systemctl enable sshd
   sudo systemctl start sshd
   
   sudo firewall-cmd --permanent --add-service=http
   sudo firewall-cmd --permanent --add-service=https
   sudo systemctl reload firewalld
   ```

3. 下载rpm安装包  选择对应的包后 点击右侧的`wget`复制  或者下载到本地通过sftp传到虚拟机上

   ```bash
   https://packages.gitlab.com/gitlab/gitlab-ce
   ```

4. rpm指令安装gitlab安装包

   ```bash
   [root @ sora-localhost ~]  rpm -ivh gitlab-ce-13.11.0-ce.0.el7.x86_64.rpm
         ...
         Please configure a URL for your GitLab instance by setting `external_url`
         configuration in /etc/gitlab/gitlab.rb file.
         Then, you can start your GitLab instance by running the following command:
         sudo gitlab-ctl reconfigure
   ```

5. 根据提示更改url地址

   ```bash
   [root @ sora-localhost ~]  vim /etc/gitlab/gitlab.rb 
         external_url 'http://10.200.9.92'   #修改为本机ip地址
   ```

6. 根据提示更改 运行服务前 输入指令 进行重新配置

   ```bash
   [root @ sora-localhost ~]  gitlab-ctl reconfigure
   ```

7. 启动Gitlab

   ```bash
   gitlab-ctl status
   ```

   





# Git常用指令

```bash
git init				#初始化仓库把一个目录初始化为版本仓库(可以是空的目录也可以是帯内容的目录) 
git status			#査看当前仓库的状态

git add <文件名> 		#添加文件到暂存区
git add ./*		#添加当前所有文件到暂存区 

git rm --cached <文件名>		#撤出暂存区
git rm -f <文件名>		#同时删除暂存区和工作目录的文件

git commit -m   "描述信息"		#从暂存区提交到本地仓库
git commit -am  "描述信息"		#如果某个文件已经被仓库管理,如果在更改此文件直接需要一条命令提交即可

git mv <old_name> <new_name>	#git命令直接更改名称

git diff								#默认比对工作目录和暂存区 
git diff	--cached			#比对暂存区和本地仓库

git log									# 查看历史快照
git log --oneline				# 一行简单的显示 Commit信息
git log --oneline --decorate			# 显示当前的指针指向哪里
git log  -p							# 显示具体内容的变化
git log  -1							# 只显示1条内容

git reset --hard  <哈希值>/<标签>			 # 回滚数据到某一个提交
git reflog 							# 可以查看所有分支的所有操作记录

git branch								# 查看分支
git branch <分支名>				# 创建分支
git branch -d <分支名>  		# 删除分支

git checkout <分支名> 							# 切换分支
git checkout -b <分支名>						# 创建并切换到分支

git merge <分支名>					#合并分支

git tag    							# 显示全部标签
git tag -a <标签名>  <哈希值> -m <说明文字>   	# 添加标签
git tag -d <标签名>			# 删除标签

git remote													# 查看远程地址
git remote add origin <SSH地址>			# 添加远程仓库地址 名称为origin
git remote rm  <本地仓库名>					# 删除远程地址
git push -u origin <master/分支名>		# 推送到远程仓库
git clone  <SSH地址>								# 从远程仓库克隆到本地仓库
```

![image-20210423145629813](/Users/jin/Documents/Typora笔记库/Pic/Git笔记/image-20210423145629813.png)



# Git常见英文

```bash
 1. branch		分支 
 2. master		主
 3. conflict  冲突
```

