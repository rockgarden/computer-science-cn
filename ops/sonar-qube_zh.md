# [使用 SonarQube 进行代码分析](https://www.baeldung.com/sonar-qube)

1. 概述

    在本文中，我们将探讨使用 SonarQube 进行静态源代码分析，这是一个确保代码质量的开源平台。

    让我们从一个核心问题开始--为什么首先要分析源代码？简单地说，就是确保项目生命周期内的质量、可靠性和可维护性。

2. 本地运行 SonarQube

    在本地运行 SonarQube 有两种可能：

    - 从压缩文件运行 Sonarqube 服务器。要下载 SonarQube 的最新 LTS 版本，请点击[此处](https://www.sonarqube.org/downloads/)。按照本[快速入门指南](https://docs.sonarqube.org/latest/setup/get-started-2-minutes/)所述设置本地服务器。
    - 在 Docker 容器中运行 Sonarqube。运行以下命令启动服务器
        `docker run -d --name sonarqube -e SONAR_ES_BOOTSTRAP_CHECKS_DISABLE=true -p 9000:9000 sonarqube:latest`

3. 在 SonarQube 中生成令牌

    按照前面的步骤启动一个新的 SonarQube 服务器后，现在该用 admin:admin 登录了（这是初始默认凭据，会要求你更改）。

    要在项目中使用该扫描仪，必须从 SonarQube 界面生成访问令牌。登录后，进入 "账户" 页面（<http://localhost:9000/account>），选择 "安全" 选项卡。在该选项卡中，您可以生成三种类型的令牌，以便在项目中使用：

    - 项目分析令牌 - 适用于项目级别
    - 全局分析令牌--这将在所有项目之间共享
    - 用户令牌 - 基于用户级别的访问权限，用户可访问哪个项目。

4. 分析源代码

    我们稍后将在分析项目时使用生成令牌。我们还需要选择项目的主要语言（Java）和构建技术（Maven）。

    让我们在 pom.xml 中定义插件：

    ```xml
    <build>
        <pluginManagement>
            <plugins>
                <plugin>
                    <groupId>org.sonarsource.scanner.maven</groupId>
                    <artifactId>sonar-maven-plugin</artifactId>
                    <version>3.4.0.905</version>
                </plugin>
            </plugins>
        </pluginManagement>
    </build>
    ```

    这里提供了插件的最新版本。现在，我们需要在项目目录根目录下执行以下命令对其进行扫描：

    ```bash
    mvn clean verify sonar:sonar -Dsonar.projectKey=PROJECT_KEY 
                                -Dsonar.projectName='PROJECT_NAME' 
                                -Dsonar.host.url=http://localhost:9000 
                                -Dsonar.token=THE_GENERATED_TOKEN
    ```

    我们需要将 project_key、project_name 和 the-generated-token 替换为第 3 步中设置的信息和 SonarQube 提供的生成访问令牌。

    本文中使用的项目可在[此处](https://github.com/eugenp/tutorials/tree/master/security-modules/cas/cas-secured-app)获得。

    我们指定了 SonarQube 服务器的主机 URL 和登录（生成令牌）作为 Maven 插件的参数。

    执行命令后，结果将出现在项目仪表板上 - <http://localhost:9000>。

    我们还可以将其他参数传递给 Maven 插件，甚至从 Web 界面进行设置；sonar.host.url、sonar.projectKey 和 sonar.sources 为必选参数，其他参数为可选参数。

    其他分析参数及其默认值请参见[此处](https://docs.sonarqube.org/latest/analysis/analysis-parameters/)。此外，请注意每种语言插件都有分析兼容源代码的规则。

5. 分析结果

    现在我们已经分析了第一个项目，可以访问 <http://localhost:9000> 的网页界面并刷新页面。

    发现的问题可以是 Bug、Vulnerability、Code Smell、Coverage 或 Duplication。每个类别都有相应的问题数量或百分比值。

    此外，问题还可以有五种不同的严重程度：阻塞、严重、主要、次要和信息。项目名称前面有一个图标，显示质量门状态--通过（绿色）或失败（红色）。

    点击项目名称将进入一个专门的仪表板，在这里我们可以更详细地了解项目的具体问题。

    我们可以在项目仪表板上查看项目代码、活动和执行管理任务，每个任务都有单独的选项卡。

    虽然有一个全局 "Issues" 选项卡，但项目仪表板上的 "Issues" 选项卡只显示相关项目的特定问题。

    问题选项卡总是显示类别、严重程度、标签以及纠正问题所需的计算工作量（时间）。

    通过问题选项卡，可以将问题分配给其他用户、对其发表评论并更改其严重程度。点击问题本身会显示有关问题的更多细节。

    问题选项卡左侧有复杂的过滤器。这些都是精确定位问题的好帮手。那么，如何才能知道代码库是否足够健康，可以部署到生产环境中呢？这就是质量门的作用。

6. SonarQube 质量门

    在本节中，我们将了解 SonarQube 的一个关键功能--质量门。然后，我们将看到一个如何设置自定义质量门的示例。

    1. 什么是质量门？

        质量门是项目在获得生产发布资格之前必须满足的一系列条件。它回答了一个问题：我是否可以在当前状态下将代码推向生产？

        确保 "new" 代码的质量，同时修复现有代码，是长期保持一个良好代码库的好方法。质量门(Quality Gate)有助于制定规则，以便在后续分析中验证添加到代码库中的每一个新代码。

        质量门中设置的条件仍会影响未修改的代码段。如果我们能防止出现新问题，随着时间的推移，我们就能消除所有问题。

        这种方法相当于从源头上[解决漏水问题](https://www.sonarsource.com/blog/water-leak-changes-the-game-for-technical-debt-management/)。这就引出了一个特殊的术语--漏水期。这是指项目两次 analyses/versions 之间的间隔期。

        如果我们在同一项目上重新运行分析，项目仪表板的概述选项卡将显示泄漏期的结果。

        在 Web 界面中，质量门选项卡是我们访问所有已定义质量门的地方。默认情况下，服务器会预装 SonarQube 方式。

        如果出现以下情况，SonarQube way 的默认配置会将代码标记为失败：

        - 新代码的覆盖率低于 80
        - 新代码的重复行百分比大于 3
        - 可维护性、可靠性或安全性评级低于 A

        有了这些了解，我们就可以创建自定义质量门。

    2. 添加自定义质量门

        首先，我们需要点击 "Quality Gates" 选项卡，然后点击页面左侧的 "Create" 按钮。我们需要给它取一个名字 - baeldung。

        现在我们可以设置所需的条件：

        从 "Add Condition" 下拉菜单中，我们选择 "Blocker Issues"，它会立即显示在条件列表中。

        我们将指定 "greater" 作为运算符，错误列设置为零（0），并选中 "Over Leak Period"列：

        创建自定义闸门 2
        然后点击 "添加" 按钮，使更改生效。让我们按照上述相同步骤添加另一个条件。

        我们将从 "添加条件" 下拉菜单中选择" 问题"，并选中 "过泄漏期" 列。

        操作符列的值将设置为"小于"，我们将在 "错误" 列中添加一(1) 作为值。这意味着，如果新增代码中的问题数小于 1，则将质量门标记为失败。

        我知道这在技术上说不通，但为了便于学习，我们还是用它吧。别忘了点击 "添加" 按钮保存规则。

        最后一步，我们需要为自定义质量门附加一个项目。我们可以通过向下滚动页面到 "项目" 部分来实现。

        我们需要点击 "全部"，然后标记我们选择的项目。我们还可以在页面右上角将其设置为默认质量门。

        我们将再次扫描项目源代码，就像之前使用 Maven 命令一样。扫描完成后，我们进入项目选项卡并刷新。

        这次，项目将不符合质量门标准，并将失败。为什么会这样？因为我们在其中一条规则中规定，如果没有新问题，项目就会失败。

        让我们回到质量门选项卡，将问题的条件改为大于。我们需要点击更新按钮来实现这一更改。

        这次对源代码的新扫描将通过。

7. 将 SonarQube 集成到 CI 中

    可以将 SonarQube 作为持续集成流程的一部分。如果代码分析不满足质量门条件，构建将自动失败。

    为了实现这一目标，我们将使用[SonarCloud](https://sonarcloud.io/)，它是SonaQube服务器的云托管版本。我们可以在[这里](https://sonarcloud.io/sessions/new)创建一个账户。

    在 "我的账户>组织" 中，我们可以看到组织密钥，其形式通常为 xxxx-github 或 xxxx-bitbucket。

    在 "我的账户>安全"中，我们还可以生成一个令牌，就像在服务器本地实例中一样。记下令牌和组织密钥，以便以后使用。

    在本文中，我们将使用 Travis CI，并使用现有的 Github 配置文件在这里创建一个账户。它将加载我们所有的项目，我们可以在任何项目上打开开关，激活 Travis CI。

    我们需要将在 SonarCloud 上生成的令牌添加到 Travis 环境变量中。点击已激活 CI 的项目即可。

    然后点击 "更多选项>设置"，再向下滚动到 "环境变量"。

    我们将添加一个名为 SONAR_TOKEN 的新条目，并使用在 SonarCloud 上生成的令牌作为值。Travis CI 将对其进行加密和隐藏，不让公众看到。

    最后，我们需要在项目根目录下添加一个 .travis.yml 文件，内容如下：

    ```yml
    language: java
    sudo: false
    install: true
    addons:
    sonarcloud:
        organization: "your_organization_key"
        token:
        secure: "$SONAR_TOKEN"
    jdk:
    - oraclejdk8
    script:
    - mvn clean org.jacoco:jacoco-maven-plugin:prepare-agent package sonar:sonar
    cache:
    directories:
        - '$HOME/.m2/repository'
        - '$HOME/.sonar/cache'
    ```

    请记住用上述组织密钥替代您的组织密钥。提交新代码并推送到 Github repo 会触发 Travis CI 编译，进而激活声纳扫描。

8. 总结

    在本教程中，我们了解了如何在本地设置 SonarQube 服务器，以及如何使用质量门（Quality Gate）定义项目是否适合生产发布的标准。

    SonarQube [文档](https://docs.sonarqube.org/latest/)中有更多关于平台其他方面的信息。
