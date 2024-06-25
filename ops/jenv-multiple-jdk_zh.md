# [使用 jEnv 管理多个 JDK 安装](https://www.baeldung.com/jenv-multiple-jdk)

1. 简介

    随着 Java 每个新版本的发布，我们可能需要在环境中管理多个并行版本的软件开发工具包（SDK）。因此，设置和管理 [JAVA_HOME](https://www.baeldung.com/java-home-on-windows-7-8-10-mac-os-x-linux) 路径变量有时会变得非常麻烦。

    在本教程中，我们将了解 jEnv 如何帮助管理多个不同版本的 JDK 安装。

2. 什么是 jEnv？

    jEnv 是一个命令行工具，可帮助我们管理多个 JDK 安装。它基本上是在我们的 shell 中设置 JAVA_HOME，设置方式可以是全局设置、本地设置到当前工作目录或按 shell 设置。

    它能让我们在不同的 Java 版本之间快速切换。这在处理具有不同 Java 版本的多个应用程序时尤其有用。

    值得注意的是，jEnv 并不为我们安装 Java JDK。相反，它只是帮助我们方便地管理多个 JDK 安装。

    接下来，让我们深入了解一下 jEnv 的安装，并回顾一下其最常用的命令。

3. 安装 jEnv

    jEnv 支持 Linux 和 MacOS 操作系统。此外，它还支持 Bash 和 Zsh shell。让我们从使用终端安装开始：

    在 MacOS 上，我们可以使用 [Homebrew](https://brew.sh/) 简单地安装 jEnv：

    `$ brew install jenv`

    在 Linux 上，我们可以从源代码安装 jEnv：

    `$ git clone https://github.com/jenv/jenv.git ~/.jenv`

    接下来，让我们根据使用的 shell 将安装的 jenv 命令添加到路径中。

    为 Bash shell 添加 PATH 条目：

    ```shell
    echo 'export PATH="$HOME/.jenv/bin:$PATH"' >> ~/.bash_profile
    echo 'eval "$(jenv init -)"' >> ~/.bash_profile
    source ~/.bash_profile
    ```

    为 Zsh shell 添加 PATH 条目：

    ```shell
    echo 'export PATH="$HOME/.jenv/bin:$PATH"' >> ~/.zshrc
    echo 'eval "$(jenv init -)"' >> ~/.zshrc
    source ~/.zshrc
    ```

    最后，我们使用 jenv doctor 命令来验证 jEnv 的安装。在 MacOS 上，该命令将显示如下内容：

    ```shell
    $ jenv doctor
    [OK] No JAVA_HOME set
    [ERROR] Java binary in path is not in the jenv shims.
    [ERROR] Please check your path, or try using /path/to/java/home is not a valid path to java installation.
    PATH : /opt/homebrew/Cellar/jenv/0.5.4/libexec/libexec:/Users/jenv/.jenv/shims:/Users/user/.jenv/bin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
    [OK] Jenv is correctly loaded
    ```

    这表明 jenv 已正确安装并加载，但 Java 尚未安装。

    接下来，让我们看看如何安装和管理多个 JDK 版本。

4. 管理 JDK 安装

    让我们从设置 JDK 版本开始。我们可以使用 [brew、yum 或 apt](https://www.baeldung.com/linux/yum-and-apt) 等可用的软件包管理器安装 JDK。或者，我们也可以下载 JDK 并将其放在某个文件夹中。

    jEnv 的好处是我们不需要通过软件包管理器安装 JDK。我们只需下载 JDK 并将其放入某个文件夹即可。

    1. 向 jEnv 添加 JDK

        首先，要在 jEnv 中使用新的 JDK，我们需要告诉 jEnv JDK 的位置。为此，我们使用 jenv add 命令并指定 JDK 的路径：

        ```shell
        $ jenv add /Library/Java/JavaVirtualMachines/openjdk-8.jdk/Contents/Home/
        openjdk8-1.8.0.332 added
        1.8.0.332 added
        1.8 added
        ```

        这将把 JDK 8 添加到 jEnv 中。每个版本都有三个不同的名称。让我们再次运行 jenv doctor 来确认 JDK 设置：

        ```shell
        $ jenv doctor
        [OK] No JAVA_HOME set
        [OK] Java binaries in path are jenv shims
        [OK] Jenv is correctly loaded
        ```

        我们可以看到，jEnv 现在能识别已配置的 JDK。

        此外，让我们使用 jenv versions 命令列出 jEnv 中所有可用的 JDK：

        ```shell
        $ jenv versions
        * system (set by /Users/user/.jenv/version)
        1.8
        1.8.0.332
        openjdk64-1.8.0.332
        ```

        这将列出在 jEnv 中注册的所有 JDK。在我们的例子中，jEnv 配置了 JDK 8。

        为了演示如何使用多个 JDK，让我们再安装一个 JDK 版本 - 11 并用 jEnv 进行配置：

        ```shell
        $ jenv add /Library/Java/JavaVirtualMachines/openjdk-11.jdk/Contents/Home/
        openjdk64-11.0.15 added
        11.0.15 added
        11.0 added
        11 added
        ```

        最后，现在运行 jenv versions 命令将同时列出已配置的 JDK 版本：

        ```shell
        $ jenv versions
        * system (set by /Users/avinb/.jenv/version)
        1.8
        1.8.0.332
        11
        11.0
        11.0.15
        openjdk64-11.0.15
        openjdk64-1.8.0.332
        ```

        显然，我们现在已经用 jEnv.4.2 配置了两个 JDK 版本。

    2. 使用 jEnv 管理 JDK 版本

        jEnv 支持三种类型的 JDK 配置：

        - Global - 如果我们在计算机上的任何地方的命令行中输入 java 命令，将使用的 JDK。
        - Local - 仅为特定文件夹配置的 JDK。在文件夹中输入 java 命令将使用本地 JDK 版本，而不是全局 JDK 版本。
        - Shell - 仅用于当前 shell 实例的 JDK。

        首先，让我们检查全局 JDK 的版本：

        ```shell
        $ jenv global
        system
        ```

        该命令输出 "system"，表示系统安装的 JDK 将被用作全局 JDK。让我们将全局 JDK 版本设为 JDK 11：

        `$ jenv global 11`

        现在检查全局版本将显示 JDK 11：

        ```shell
        $ jenv global 
        11
        ```

        接下来，让我们看看如何设置本地 JDK 版本。

        例如，我们在 ~/baeldung-project 目录中有一个使用 JDK 8 的示例项目。让我们 cd 进入此目录，检查此项目的本地 JDK 版本：

        ```shell
        $ jenv local
        jenv: no local version configured for this directory
        ```

        这条错误信息表明，我们尚未为该目录设置任何本地 JDK 版本。在没有本地 JDK 版本的情况下运行 jenv version 命令将显示全局 JDK 版本。让我们为该目录设置一个本地 JDK 版本：

        `$ jenv local 1.8`

        该命令将在 ~/baeldung-project 目录中设置本地 JDK。设置本地 JDK 基本上是在当前目录下创建一个名为 .java-version 的文件。该文件包含我们设置的本地 JDK 版本 "1.8"。

        在该目录下再次运行 jenv version 命令，现在将输出 JDK 8。让我们检查一下在此目录中设置的本地 JDK 版本：

        ```shell
        $ jenv local
        1.8
        ```

        最后，要为某个 shell 实例设置 JDK 版本，我们使用 jenv shell 命令：

        `$ jenv shell 1.8`

        这将设置当前 shell 实例的 JDK 版本，并覆盖已设置的本地和全局 JDK 版本。

    3. 使用 Maven 和 Gradle 配置 jEnv

        众所周知，[Maven](https://www.baeldung.com/maven) 和 [Gradle](https://www.baeldung.com/gradle) 等工具使用系统 JDK 运行。它不会使用 jEnv 配置的 JDK。为了确保 jEnv 能与 Maven 和 Gradle 正常工作，我们必须启用它们各自的插件。

        对于 Maven，我们将启用 jEnv maven 插件：

        `$ jenv enable-plugin maven`

        同样，对于 Gradle，我们将启用 jEnv gradle 插件：

        ```shell
        jenv enable-plugin gradle
        gradle plugin activated
        jenv: cannot rehash: /Users/wangkan/.jenv/shims/.jenv-shim exists
        ```

        现在运行 Maven 和 Gradle 命令将使用特定于 jEnv 的 JDK 版本，而不是系统 JDK。

        请注意，有时 jEnv 可能不会选择正确的 JDK 版本，因此我们可能会出错。在这种情况下，我们可能需要启用 jEnv 导出插件：

        `$ jenv enable-plugin export`

        换句话说，该插件将确保 JAVA_HOME 变量设置正确。

        此外，除其他工具外，[SDKMAN](https://www.baeldung.com/java-sdkman-intro) 也是一个管理 JDK 的替代工具。

5. 结论

    在本文中，我们首先了解了什么是 jEnv 以及如何安装它。

    然后，我们了解了 jEnv 如何帮助我们方便地配置和管理不同的 JDK 安装。接着，我们了解了如何使用 jEnv 快速使用全局、本地和特定于 shell 的 JDK 版本。这对我们处理具有不同 JDK 版本的多个不同项目尤其有帮助。

    最后，我们了解了如何配置 jEnv 以与 Maven 和 Gradle 等构建工具配合使用。
