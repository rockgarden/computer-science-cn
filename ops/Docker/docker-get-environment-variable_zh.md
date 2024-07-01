# [从 Docker 容器获取环境变量](https://www.baeldung.com/ops/docker-get-environment-variable)

1. 概述

    Docker 是一个容器化平台，可将应用程序及其所有依赖项打包。理想情况下，这些应用程序需要一定的环境才能启动。在 Linux 中，我们使用环境变量来满足这一要求。这些变量决定了应用程序的行为。

    在本教程中，我们将学习如何检索运行 Docker 容器时设置的所有环境变量。向 Docker 容器传递环境变量有多种方法，而获取已设置的这些变量也有不同方法。

    在进一步学习之前，让我们先了解一下环境变量的必要性。

2. 了解 Linux 中的环境变量

    环境变量是一组动态的键值对，可在全系统范围内访问。这些变量可以帮助系统定位软件包、配置任何服务器的行为，甚至使 bash 终端输出直观。

    默认情况下，主机上的环境变量不会传递给 Docker 容器。原因是 Docker 容器应该与主机环境隔离。因此，如果我们想在 Docker 容器中使用环境，就必须明确地设置它。

    现在，让我们来看看从 Docker 容器内部获取环境变量的不同方法。

3. 使用 docker exec 命令获取

    为了演示的目的，让我们先运行一个 Alpine Docker 容器，并向其传递一些环境变量：

    ```shell
    docker run -itd --env "my_env_var=baeldung" --name mycontainer alpine
    9de9045b5264d2de737a7ec6ba23c754f034ff4f35746317aeefcea605d46e84
    ```

    在这里，我们在名为 mycontainer 的 Docker 容器中传递了 my_env_var 的值 baeldung。

    现在，让我们使用 [docker exec](https://docs.docker.com/engine/reference/commandline/exec/) 命令来获取名为 my_env_var 的环境变量：

    ```shell
    $ docker exec mycontainer /usr/bin/env
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    HOSTNAME=9de9045b5264
    my_env_var=baeldung
    HOME=/root
    ```

    > /usr/bin/env 可简写为 env，即 `docker exec mycontainer env`

    在这里，我们在 Docker 容器内执行 /usr/bin/env 实用程序。使用该实用程序，你可以查看 Docker 容器内设置的所有环境变量。请注意，我们的 my_env_var 也出现在输出中。

    我们还可以使用以下命令来获得类似的结果：

    ```shell
    $ docker exec mycontainer /bin/sh -c /usr/bin/env
    HOSTNAME=9de9045b5264
    SHLVL=1
    HOME=/root
    my_env_var=baeldung
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    PWD=/
    ```

    注意，与之前的输出相比，环境变量更多了。这是因为这次我们是在 /bin/sh 二进制文件的帮助下执行命令的。该二进制文件会隐式设置一些额外的环境变量。

    此外，/bin/sh shell 并不是所有 Docker 镜像中都必须存在的。例如，在包含 /bin/bash shell 的 centos Docker 镜像中，我们将使用以下命令获取环境变量：

    ```shell
    $ docker run -itd --env "container_type=centos" --name centos_container centos
    aee6f2718f18723906f7ab18ab9c37a539b6b2c737f588be71c56709948de9eb
    $ docker exec centos_container bash -c /usr/bin/env
    container_type=centos
    HOSTNAME=aee6f2718f18
    PWD=/
    HOME=/root
    SHLVL=1
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    _=/usr/bin/env
    ```

    我们还可以使用 docker exec 命令获取单个环境变量的值：

    `$ docker exec mycontainer printenv my_env_var`

    示例：`docker exec datahub-datahub-frontend-react-1 printenv AUTH_OIDC_USER_NAME_CLAIM`

    [printenv](https://man7.org/linux/man-pages/man1/printenv.1.html) 是另一个显示 Linux 环境变量的命令行工具。在这里，我们将环境变量的名称 my_env_var 作为参数传递给 printenv。这将打印出 my_env_var 的值。

    这种方法的缺点是，要检索环境变量，Docker 容器必须处于运行状态。

4. 使用 docker inspect 命令获取

    现在，让我们来看看另一种在 Docker 容器处于停止状态时获取环境变量的方法。为此，我们将使用 docker inspect 命令。

    [docker inspect](https://docs.docker.com/engine/reference/commandline/inspect/) 提供了所有 Docker 资源的详细信息。输出是 JSON 格式的。因此，我们可以根据需要过滤输出。

    让我们操纵 docker inspect 命令，只显示容器的环境变量：

    ```shell
    $ docker inspect mycontainer --format "{{.Config.Env}}"
    [my_env_var=baeldung PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin]
    ```

    在这里，我们使用 -format 选项过滤了 docker inspect 输出中的环境变量。同样，输出中出现了 my_env_var。

    使用 docker inspect 命令检查单个环境变量：

    ```shell
    $ docker inspect mycontainer | jq -r '.[].Config.Env[]|select(match("^my_env_var"))|.[index("=")+1:]'
    baeldung
    ```

    [jq](https://www.baeldung.com/linux/jq-command-json) 是一个轻量级 JSON 处理器，可以解析和转换 JSON 数据。在这里，我们将 docker inspect 的 JSON 输出传递给 jq 命令。然后，它会搜索 my_env_var 变量，并通过在"="上分割显示其值。

    请注意，我们还可以在 docker exec 和 docker inspect 命令中使用容器 id。

    与 docker exec 不同，docker inspect 命令对停止和运行的容器都有效。

5. 总结

    在本文中，我们学习了如何从 Docker 容器中获取所有环境变量。我们首先讨论了环境变量在 Linux 中的重要性。然后，我们学习了检索环境变量的 docker exec 和 docker inspect 命令。

    docker exec 方法有一些限制，而 docker inspect 命令在任何情况下都能运行。
