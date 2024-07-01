# [向Docker容器传递环境变量](https://www.baeldung.com/ops/docker-container-environment-variables)

1. 概述

    将我们的服务与其配置分开通常是个好主意。对于[十二要素应用程序](https://www.baeldung.com/spring-boot-12-factor)，我们应该将配置存储在环境中。

    当然，这意味着我们需要一种将配置注入服务的方法。

    在本教程中，我们将通过向 Docker 容器传递环境变量来实现这一点。

2. 使用 -env、-e

    在本教程中，我们将使用名为 Alpine 的小型（5MB）Linux 映像。让我们从本地调用该镜像开始：

    `docker pull alpine:3`

    当我们启动 Docker 容器时，可以使用参数 -env （或其简写 -e）将环境变量作为键值对直接传入命令行。

    例如，让我们执行以下命令：

    `$ docker run --env VARIABLE1=foobar alpine:3 env`

    简单地说，就是把我们设置的环境变量反映到控制台：

    `VARIABLE1=foobar`

    可以看到，Docker 容器能正确理解变量 VARIABLE1。

    此外，如果变量已经存在于本地环境中，我们还可以省略命令行中的值。

    例如，让我们定义一个本地环境变量：

    `$ export VARIABLE2=foobar2`

    然后，我们来指定这个环境变量，但不去掉它的值：

    `docker run --env VARIABLE2 alpine:3 env`

    我们可以看到，Docker 还是从周围的环境中获取了变量值：

    `VARIABLE2=foobar2`

3. 使用 -env 文件

    当变量数量较少时，上述解决方案就足够了。但是，一旦变量数量超过少数，上述方法就会变得非常麻烦，而且容易出错。

    另一种解决方案是使用标准的 key=value 格式，用文本文件来存储变量。

    让我们在一个名为 my-env.txt 的文件中定义几个变量：

    ```shell
    echo VARIABLE1=foobar1 > my-env.txt
    echo VARIABLE2=foobar2 >> my-env.txt
    echo VARIABLE3=foobar3 >> my-env.txt
    ```

    现在，让我们把这个文件注入 Docker 容器：

    `$ docker run --env-file my-env.txt alpine:3 env`

    最后，让我们看看输出结果：

    ```log
    VARIABLE1=foobar1
    VARIABLE2=foobar2
    VARIABLE3=foobar3
    ```

4. 使用 Docker Compose

    Docker Compose 还提供了定义环境变量的功能。对这一特定主题感兴趣的人，可以查看我们的 [Docker Compose 教程](https://www.baeldung.com/docker-compose#managing-environment-variables)，了解更多详情。

5. 小心敏感值

    通常情况下，变量之一是数据库或外部服务的密码。我们必须谨慎处理如何将这些变量注入 Docker 容器。

    直接通过命令行传递这些值可能是最不安全的做法，因为在我们意想不到的地方（如源控制系统或操作系统进程列表）泄漏敏感值的风险更大。

    在本地环境或文件中定义敏感值是更好的选择，因为两者都可以防止未经授权的访问。

    不过，重要的是要意识到，任何拥有 Docker 运行时访问权限的用户都可以检查运行中的容器并发现秘密值。

    让我们检查一个正在运行的容器

    `docker inspect 6b6b033a3240`

    输出结果显示了环境变量：

    ```json
    "Config": {
        // ...
        "Env": [
        "VARIABLE1=foobar1",
        "VARIABLE2=foobar2",
        "VARIABLE3=foobar3",
        // ...
        ]
    }
    ```

    对于那些担心安全性的情况，有必要提一下 Docker 提供的一种叫做 Docker Secrets 的机制。像 Kubernetes、AWS 或 Azure 提供的容器服务也提供类似的功能。

6. 总结

    在这个简短的教程中，我们了解了向 Docker 容器注入环境变量的几种不同方案。

    虽然每种方法都很有效，但我们的选择最终将取决于各种参数，如安全性和可维护性。
