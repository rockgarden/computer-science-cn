# [SDKMAN 指南！](https://www.baeldung.com/java-sdkman-intro)

1. 概述

    随着 Java 新版本周期的到来，开发人员可能需要在其环境中管理并行版本和不同版本的软件开发工具包 (SDK)。因此，设置 PATH 变量有时会变得非常麻烦。

    在本教程中，我们将了解 SDKMAN! 如何帮助轻松管理 SDK 的安装和选择。

2. 什么是 SDKMAN！？

    SDKMAN! 是一款管理多个 SDK 并行版本的工具，SDKMAN! 将其称为 "候选(candidates)"。

    它提供了方便的命令行界面（CLI）和 API，用于列出、安装、切换和删除候选版本。此外，它还能为我们设置环境变量。

    它还允许开发人员安装基于 JVM 的 SDK，如 Java、Groovy、Scala、Kotlin 和 Ceylon。此外还支持 Maven、Gradle、SBT、Spring Boot、Vert.x 等。SDKMAN! 是一款用 Bash 编写的免费、轻量级开源工具。

3. 安装 SDKMAN！

    所有主流操作系统都支持 SDKMAN！，所有基于 Unix 的系统都能轻松安装。此外，它还支持 Bash 和 Zsh shell。

    因此，让我们先用终端安装它：

    `$ curl -s "https://get.sdkman.io" | bash`

    然后，按照屏幕上的提示完成安装。

    我们可能需要安装 zip 和 unzip 软件包来完成安装过程。

    接下来，打开一个新的终端或运行

    `$ source "$HOME/.sdkman/bin/sdkman-init.sh"`

    最后，运行以下命令确保安装成功。如果一切顺利，则应显示版本：

    ```shell
    $ sdk version
    SDKMAN 5.8.5+522
    ```

    有关更多定制信息，请参阅 SDKMAN! 网站上的[安装指南](https://sdkman.io/install)。

    要查看所有可用命令，请使用帮助命令：

    `$ sdk help`

4. 列出所有 SDK 候选程序

    首先，让我们列出所有可用的 SDK 候选程序。

    `$ sdk list`

    list 命令会显示所有可用的候选 SDK，这些候选 SDK 由唯一名称、描述、官方网站和安装命令标识：

    ```shell
    =====================================================
    Available Candidates
    =====================================================
    q-quit                                  /-search down
    j-down                                  ?-search up
    k-up                                    h-help
    -----------------------------------------------------
    Java (11.0.7.hs-adpt)                https://zulu.org
    ...
                                    $ sdk install java
    -----------------------------------------------------
    Maven (3.6.3)                https://maven.apache.org
    ...
                                    $ sdk install maven
    -----------------------------------------------------
    Spring Boot (2.3.1.RELEASE)          http://spring.io
    ...
                                $ sdk install springboot
    ------------------------------------------------------
    ...
    ```

    因此，我们可以使用此标识符安装 Spring Boot (2.3.1.RELEASE) 或 Maven (3.6.3) 等候选版本的默认版本。此列表中指定的版本代表每个 SDK 的稳定版或 LTS 版本。

5. 安装和管理 Java 版本

    1. 列出版本

        要列出 Java 的可用版本，请使用 list 命令。结果是一个按供应商分组并按版本排序的条目表：

        ```shell
        $ sdk list java
        ===================================================================
        Available Java Versions
        ===================================================================
        Vendor       | Use | Version | Dist    | Status | Identifier
        -------------------------------------------------------------------
        AdoptOpenJDK |     | 14.0.1  | adpt    |        | 14.0.1.j9-adpt
        ...
        Amazon       |     | 11.0.8  | amzn    |        | 11.0.8-amzn
        ...
        Azul Zulu    |     | 14.0.2  | zulu    |        | 14.0.2-zulu
        ...
        BellSoft     |     | 14.0.2  | librca  |        | 14.0.2.fx-librca
        ...
        GraalVM      |     | 20.1.0  | grl     |        | 20.1.0.r11-grl
        ...
        Java.net     |     | 16.ea   | open    |        | 16.ea.6-open
        ...
        SAP          |     | 14.0.2  | sapmchn |        | 14.0.2-sapmchn
        ...
        ```

        每次要检查、切换或管理候选程序的存储时，我们都需要使用这条命令。

    2. 安装 Java 版本

        假设我们要从 Azul Zulu 安装 Java 14 的最新版本。因此，我们要复制它的标识符，即表格中的版本，并将其作为参数添加到安装命令中：

        ```shell
        $ sdk install java 14.0.2-zulu
        Downloading: java 14.0.2-zulu
        In progress...
        ########### 100.0%
        Repackaging Java 14.0.2-zulu...
        Done repackaging...
        Installing: java 14.0.2-zulu
        Done installing!
        Setting java 14.0.2-zulu as default.
        ```

        SDKMAN! 将下载并解压该版本到我们计算机上的一个目录中。

        此外，它还会更新环境变量，以便我们能立即在终端中使用 Java。

        我们可以使用 list 命令来验证任何版本的状态和使用情况。因此，现在已安装并正在使用 14.0.1 版本：

        ```shell
        $ sdk list java
        =================================================================
        Available Java Versions
        =================================================================
        Vendor    | Use | Version | Dist    | Status    | Identifier
        -----------------------------------------------------------------
        ...
        Azul Zulu | >>> | 14.0.1  | adpt    | installed | 14.0.1.j9-adpt
        ...
        ```

        此外，还可以使用相同的命令从计算机上安装 Java 或任何自定义版本，但要指定二进制文件的路径作为附加参数：

        `$ sdk install java custom-8 ~/Downloads/my-company-jdk-custom-8`

    3. 版本间切换

        我们可以通过两种形式控制版本间的切换，暂时如下

        `$ sdk use java 14.0.1.j9-adpt`

        或永久

        `$ sdk default java 14.0.1.j9-adpt`

    4. 删除版本

        要删除已安装的版本，请使用目标版本运行卸载命令：

        `$ sdk uninstall java 14.0.1.j9-adpt`

    5. 显示使用中的版本

        要检查 Java 的当前版本，我们可以运行 current 命令：

        ```shell
        $ sdk current java
        Using java version 14.0.2-zulu
        ```

        同样，最后一条命令也有相同的效果：

        `$ java -version`

        要按 SDK 显示机器上的版本，我们可以不带参数运行当前命令：

        ```shell
        $ sdk current
        Using:
        java: 14.0.2-zulu
        gradle: 6.2.2
        ```

6. 与集成开发环境一起使用 SDKMAN!

    安装的 SDK 保存在 SDKMAN! 目录中，默认为 ~/.sdkman/candidates。

    例如，不同版本的 Java 也可在 ~/.sdkman/candidates/java/ 目录下找到，子目录以版本命名：

    ```shell
    $ ls -al ~/.sdkman/candidates/java/
    total 0
    drwxrwxrwx 1 user user 12 Jul 25 20:00 .
    drwxrwxrwx 1 user user 12 Jul 25 20:00 ..
    drwxrwxr-x 1 user user 12 Jul 25 20:00 14.0.2-zulu
    lrwxrwxrwx 1 user user 14 Jul 25 20:00 current -> 14.0.2-zulu
    ```

    因此，当前选择的 Java 版本也将作为 current 出现在该目录中。

    同样，Gradle 或其他 SDK 也将安装在 candidates 目录下。

    这样，我们就可以在自己喜欢的集成开发环境中使用任何特定版本的 Java。我们只需复制特定版本的路径，并在集成开发环境的配置中进行设置即可。

    1. IntelliJ IDEA

        在 IntelliJ IDEA 中，打开 "Project Structure（项目结构）"，然后打开 "Project Settings（项目设置）"。在项目配置中，我们可以从 "Project SDK"（项目 SDK）部分选择 "New..."（新建...）来添加新的 Java 版本。

        我们还可以在 "Build Tools（构建工具）"部分定义要使用的 Java、Gradle 或 Maven 版本。

        提示：Java 版本必须与 Gradle 或 Maven 的 "项目 SDK "中使用的版本相同。

    2. Eclipse

        在 Eclipse 中打开 "Project Properties"（项目属性），选择 "Java Build Path"（Java 构建路径），然后切换到 "Libraries"（库）选项卡。在这里，我们可以通过 "Add Library...（添加库...）"并按照说明管理新的 Java SDK。

        我们还可以控制所有项目已安装的 SDK。打开 "Window"（窗口）菜单下的 "Preferences"（偏好设置），然后转到 "Installalled JREs"（已安装的 JREs）。在这里，我们可以通过 "Add...（添加...）"并按照说明管理 Java 的 SDK：

7. 总结

    在本教程中，我们展示了 SDKMAN! 如何帮助我们在 Maven 等其他 Java 环境工具中管理不同版本的 Java SDK。
