# [Docker 日志指南](https://www.baeldung.com/ops/docker-logs)

1. 概述

    [Docker](https://www.baeldung.com/ops/docker-guide) 是一个操作系统级虚拟化平台，允许我们在容器中托管应用程序。此外，它还有助于将应用程序和基础架构分离，从而实现软件的快速交付。

    [Docker 容器](https://docs.docker.com/engine/reference/commandline/container/)生成的日志文件包含各种有用信息。每当事件发生时，Docker 容器都会创建日志文件。

    Docker 会将日志生成到 STDOUT 或 STDERR，其中包括日志来源、输出流数据和时间戳。调试和查找问题的根本原因可以通过日志文件来完成。

    在本教程中，我们将研究以不同方式访问 Docker 日志。

2. 了解 Docker 日志

    在 Docker 中，主要有两种类型的日志文件。Docker 守护进程日志提供了对 Docker 服务整体状态的深入了解。Docker 容器日志涵盖与特定容器相关的所有日志。

    我们将主要探讨访问 Docker 容器日志的不同命令。我们将使用 [docker logs](https://docs.docker.com/engine/reference/commandline/logs/) 命令和直接访问系统上的日志文件来检查容器日志。

    日志文件对调试问题非常有用，因为它们提供了发生问题的详细信息。通过分析 Docker 日志，我们可以更快地诊断和排除故障。

3. 使用 docker 日志命令

    在继续之前，让我们先运行一个 Postgres Docker 容器示例：

    ```shell
    $ docker run -itd -e POSTGRES_USER=baeldung -e POSTGRES_PASSWORD=baeldung -p 5432:5432 -v /data:/var/lib/postgresql/data --name postgresql-baeldung postgres
    Unable to find image 'postgres:latest' locally
    latest: Pulling from library/postgres
    214ca5fb9032: Pull complete 
    ...
    Status: Downloaded newer image for postgres:latest
    bce34bb3c6175fe92c50d6e5c8d2045062c2b502b9593a258ceb6cafc9a2356a
    ```

    为了说明问题，让我们查看一下 postgresql-baeldung 容器的 containerId：

    ```log
    $ docker ps
    CONTAINER ID        IMAGE               COMMAND                  CREATED             STATUS              PORTS                    NAMES
    bce34bb3c617        postgres            "docker-entrypoint.s…"   12 seconds ago      Up 10 seconds       0.0.0.0:5432->5432/tcp   postgresql-baeldung
    ```

    从上面命令的输出中我们可以看到，postgresql-baeldung 正在运行，其容器Id为 "bce34bb3c617"。现在，让我们使用 docker logs 命令来监控日志：

    ```shell
    $ docker logs bce34bb3c617
    2022-05-16 18:13:58.868 UTC [1] LOG:  starting PostgreSQL 14.2 (Debian 14.2-1.pgdg110+1)
    on x86_64-pc-linux-gnu, compiled by gcc (Debian 10.2.1-6) 10.2.1 20210110, 64-bit
    ...
    ```

    这里的日志包含带有时间戳的输出流数据。上述命令不包含连续日志输出。要查看容器的连续日志输出，我们需要在 docker logs 命令中使用"-follow"选项。

    "-follow"选项是最有用的 Docker 选项之一，因为它允许我们监控容器的实时日志：

    ```log
    $ docker logs --follow  bce34bb3c617
    2022-05-16 18:13:58.868 UTC [1] LOG:  starting PostgreSQL 14.2 (Debian 14.2-1.pgdg110+1)
    on x86_64-pc-linux-gnu, compiled by gcc (Debian 10.2.1-6) 10.2.1 20210110, 64-bit
    ...
    ```

    上述命令的一个缺点是，它将包含从一开始的所有日志。让我们来看看查看最近记录的连续日志输出的命令：

    ```log
    $ docker logs --follow --tail 1 bce34bb3c617
    2022-05-16 18:13:59.018 UTC [1] LOG:  database system is ready to accept connections
    ```

    我们还可以在 docker 日志命令中使用 "since" 选项来查看特定时间的文件：

    ```log
    $ docker logs --since 2022-05-16  bce34bb3c617
    2022-05-16 18:13:58.868 UTC [1] LOG:  starting PostgreSQL 14.2 (Debian 14.2-1.pgdg110+1)
    on x86_64-pc-linux-gnu, compiled by gcc (Debian 10.2.1-6) 10.2.1 20210110, 64-bit
    ...
    ```

    另外，我们也可以使用 [docker container logs](https://docs.docker.com/engine/reference/commandline/container_logs/) 命令来代替 docker logs 命令：

    ```log
    $ docker container logs --since 2022-05-16  bce34bb3c617
    2022-05-16 18:13:58.868 UTC [1] LOG:  starting PostgreSQL 14.2 (Debian 14.2-1.pgdg110+1)
    on x86_64-pc-linux-gnu, compiled by gcc (Debian 10.2.1-6) 10.2.1 20210110, 64-bit
    ...
    ```

    从上面的输出中，我们可以看到两个命令的工作原理完全相同。在新版本中，docker 容器日志命令已被弃用。

4. 使用默认日志文件

    Docker 会以 JSON 格式存储所有 STDOUT 和 STDERR 输出。此外，还可以从主机监控所有实时 Docker 日志。默认情况下，Docker 会使用 json 文件日志驱动程序将日志文件存储在主机上的一个专用目录中。日志文件目录是容器运行所在主机上的 `/var/lib/docker/containers/<container_id>`。

    为了演示，让我们查看一下 postgress-baeldung 容器的日志文件：

    ```log
    $ cat /var/lib/docker/containers/bce34bb3c6175fe92c50d6e5c8d2045062c2b502b9593a258ceb6cafc9a2356a/
    bce34bb3c6175fe92c50d6e5c8d2045062c2b502b9593a258ceb6cafc9a2356a-json.log 
    {"log":"\r\n","stream":"stdout","time":"2022-05-16T18:13:58.833312658Z"}
    ...
    ```

    在上述输出中，我们可以看到数据是 JSON 格式的。

5. 清除日志文件

    有时，我们会发现系统磁盘空间不足，Docker 日志文件占用了大量空间。为此，我们首先需要找到日志文件，然后将其删除。此外，还要确保清除日志文件不会影响正在运行的容器的状态。

    下面是清除存储在主机上的所有日志文件的命令：

    `$ truncate -s 0 /var/lib/docker/containers/*/*-json.log`

    请注意，上述命令不会删除日志文件。相反，它会删除日志文件中的所有内容。通过执行下面的命令，我们可以删除与特定容器关联的日志文件：

    `$ truncate -s 0 /var/lib/docker/containers/dd207f11ebf083f97355be1ae18420427dd2e80b061a7bf6fb0afc326ad04b10/*-json.log`

    在容器启动时，我们还可以使用 docker run 命令中的"-log-opt max-size"和"-log-opt max-file"选项从外部限制日志文件的大小：

    ```shell
    $ docker run --log-opt max-size=1k --log-opt max-file=5 -itd -e POSTGRES_USER=baeldung -e POSTGRES_PASSWORD=baeldung -p 5432:5432
    -v /data:/var/lib/postgresql/data --name postgresql-baeldung postgres
    3eec82654fe6c6ffa579752cc9d1fa034dc34b5533b8672ebe7778449726da32
    ```

    现在，让我们检查 /var/lib/docker/containers/3eec82654fe6c6ffa579752cc9d1fa034dc34b5533b8672ebe7778449726da32 目录中的日志文件数量和日志文件大小：

    ```shell
    $ ls -la
    total 68
    drwx------. 4 root root 4096 May 17 02:06 .
    drwx------. 5 root root  222 May 17 02:07 ..
    drwx------. 2 root root    6 May 17 02:02 checkpoints
    -rw-------. 1 root root 3144 May 17 02:02 config.v2.json
    -rw-r-----. 1 root root  587 May 17 02:06 3eec82654fe6c6ffa579752cc9d1fa034dc34b5533b8672ebe7778449726da32-json.log
    -rw-r-----. 1 root root 1022 May 17 02:06 3eec82654fe6c6ffa579752cc9d1fa034dc34b5533b8672ebe7778449726da32-json.log.1
    -rw-r-----. 1 root root 1061 May 17 02:06 3eec82654fe6c6ffa579752cc9d1fa034dc34b5533b8672ebe7778449726da32-json.log.2
    -rw-r-----. 1 root root 1056 May 17 02:06 3eec82654fe6c6ffa579752cc9d1fa034dc34b5533b8672ebe7778449726da32-json.log.3
    -rw-r-----. 1 root root 1058 May 17 02:06 3eec82654fe6c6ffa579752cc9d1fa034dc34b5533b8672ebe7778449726da32-json.log.4
    -rw-r--r--. 1 root root 1501 May 17 02:02 hostconfig.json
    -rw-r--r--. 1 root root   13 May 17 02:02 hostname
    -rw-r--r--. 1 root root  174 May 17 02:02 hosts
    drwx------. 2 root root    6 May 17 02:02 mounts
    -rw-r--r--. 1 root root   69 May 17 02:02 resolv.conf
    -rw-r--r--. 1 root root   71 May 17 02:02 resolv.conf.hash
    ```

    在这里，我们可以看到创建了五个日志文件，每个日志文件的最大大小为 1 kb。如果删除某些日志文件，我们将以相同的日志文件名生成新日志。

    我们还可以在 /etc/docker/daemon.json 文件中配置日志的最大大小和最大文件。让我们来看看 daemon.json 文件的配置：

    ```json
    {
        "log-driver": "json-file",
        "log-opts": {
            "max-size": "1k",
            "max-file": "5"
        }
    }
    ```

    在这里，我们在 daemon.json 中提供了相同的配置，重要的是，所有新容器都将使用此配置运行。更新 daemon.json 文件后，我们需要重新启动 Docker 服务。

6. 将 Docker 容器日志重定向到一个文件

    默认情况下，Docker 容器日志文件存储在 `/var/lib/docker/containers/<containerId>` 目录中。此外，我们还可以将 Docker 容器日志重定向到其他文件。

    为了说明这一点，让我们看看重定向容器日志的命令：

    `$ docker logs -f containername &> baeldung-postgress.log &`

    在上述命令中，我们将所有实时日志重定向到 baeldung-postgress.log 文件。此外，我们使用 & 在后台运行此命令，因此它会一直运行，直到显式停止。

7. 结束语

    在本教程中，我们学习了监控容器日志的不同方法。首先，我们查看了 docker 日志，并使用 docker 容器日志命令来监控实时日志。随后，我们使用默认的容器日志文件来监控日志。

    最后，我们研究了日志文件的清除和重定向。简而言之，我们研究了日志文件的监控和截断。
