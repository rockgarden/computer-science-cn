# [使用 Shell 命令列出 Kafka 集群中的活动代理](https://www.baeldung.com/ops/kafka-list-active-brokers-in-cluster)

1. 概述

    在监控使用 Apache Kafka 集群的事件驱动系统时，我们经常需要获取活动代理列表。在本教程中，我们将探讨几条 shell 命令，以获取正在运行的集群中的活动代理列表。

2. 设置

    在本文中，我们使用下面的 docker-compose.yml 文件来设置一个双节点的 Kafka 集群：

    ```bash
    $ cat docker-compose.yml
    ---
    version: '2'
    services:
    zookeeper-1:
        image: confluentinc/cp-zookeeper:latest
        environment:
        ZOOKEEPER_CLIENT_PORT: 2181
        ZOOKEEPER_TICK_TIME: 2000
        ports:
        - 2181:2181
    
    kafka-1:
        image: confluentinc/cp-kafka:latest
        depends_on:
        - zookeeper-1
        ports:
        - 29092:29092
        environment:
        KAFKA_BROKER_ID: 1
        KAFKA_ZOOKEEPER_CONNECT: zookeeper-1:2181
        KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka-1:9092,PLAINTEXT_HOST://localhost:29092
        KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
        KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
        KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    kafka-2:
        image: confluentinc/cp-kafka:latest
        depends_on:
        - zookeeper-1
        ports:
        - 39092:39092
        environment:
        KAFKA_BROKER_ID: 2
        KAFKA_ZOOKEEPER_CONNECT: zookeeper-1:2181
        KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka-2:9092,PLAINTEXT_HOST://localhost:39092
        KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
        KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
        KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    ```

    现在，让我们使用 [docker-compose](https://www.baeldung.com/ops/docker-compose) 命令来启动 Kafka 集群：

    `$ docker-compose up -d`

    我们可以确认 Zookeeper 服务器监听的端口是 2181，而 Kafka 代理分别监听的端口是 29092 和 39092：

    ```bash
    $ ports=(2181 29092 39092)
    $ for port in $ports
    do
    nc -z localhost $port
    done
    Connection to localhost port 2181 [tcp/eforward] succeeded!
    Connection to localhost port 29092 [tcp/*] succeeded!
    Connection to localhost port 39092 [tcp/*] succeeded!
    ```

3. 使用 Zookeeper API

    在 Kafka 集群中，[Zookeeper服务器](https://www.baeldung.com/java-zookeeper) 存储与 Kafka 代理服务器相关的元数据。因此，让我们使用 Zookeeper 公开的文件系统 API 来获取代理的详细信息。

    1. zookeeper-shell 命令

        大多数 Kafka 发行版都附带了 zookeeper-shell 或 zookeeper-shell.sh 二进制文件。因此，使用此二进制文件与 Zookeeper 服务器交互已成为事实上的标准。

        首先，让我们连接运行于 localhost:2181 的 Zookeeper 服务器：

        ```bash
        $ /usr/local/bin/zookeeper-shell localhost:2181
        Connecting to localhost:2181
        Welcome to ZooKeeper!
        ```

        连接到 Zookeeper 服务器后，我们就可以执行典型的文件系统命令（如 ls），以获取存储在服务器中的元数据信息。让我们查找当前存活的代理的 ID：

        ```bash
        ls /brokers/ids
        [1, 2]
        ```

        我们可以看到，目前有两个活动的经纪商，id 分别为 1 和 2。使用 get 命令，我们可以获取给定 id 的特定经纪人的更多详细信息：

        ```bash
        get /brokers/ids/1
        {"features":{},"listener_security_protocol_map":{"PLAINTEXT":"PLAINTEXT","PLAINTEXT_HOST":"PLAINTEXT"},"endpoints":["PLAINTEXT://kafka-1:9092","PLAINTEXT_HOST://localhost:29092"],"jmx_port":-1,"port":9092,"host":"kafka-1","version":5,"timestamp":"1625336133848"}
        get /brokers/ids/2
        {"features":{},"listener_security_protocol_map":{"PLAINTEXT":"PLAINTEXT","PLAINTEXT_HOST":"PLAINTEXT"},"endpoints":["PLAINTEXT://kafka-2:9092","PLAINTEXT_HOST://localhost:39092"],"jmx_port":-1,"port":9092,"host":"kafka-2","version":5,"timestamp":"1625336133967"}
        ```

        请注意，id=1 的代理监听端口为 29092，而 id=2 的第二个代理监听端口为 39092。

        最后，要退出 Zookeeper shell，我们可以使用 quit 命令：

        `quit`

    2. zkCli 命令

        就像 Kafka 发行版附带 zookeeper-shell 二进制文件一样，Zookeeper 发行版也附带 zkCli 或 zkCli.sh 二进制文件。

        因此，与 zkCli 的交互方式与与 zookeeper-shell 的交互方式完全相同，所以让我们继续并确认我们能够获取 id=1 的代理所需的详细信息：

        ```bash
        $ zkCli -server localhost:2181 get /brokers/ids/1
        Connecting to localhost:2181

        WATCHER::

        WatchedEvent state:SyncConnected type:None path:null
        {"features":{},"listener_security_protocol_map":{"PLAINTEXT":"PLAINTEXT","PLAINTEXT_HOST":"PLAINTEXT"},"endpoints":["PLAINTEXT://kafka-1:9092","PLAINTEXT_HOST://localhost:29092"],"jmx_port":-1,"port":9092,"host":"kafka-1","version":5,"timestamp":"1625336133848"}
        ```

        不出所料，我们可以看到使用 zookeeper-shell 获取的代理详细信息与使用 zkCli 获取的一致。

4. 使用代理版本 API

    有时，我们可能会有一个不完整的活动代理列表，我们希望获取集群中所有可用的代理。在这种情况下，我们可以使用 Kafka 发行版附带的 kafka-broker-api-versions 命令。

    假设我们知道有一个运行在 localhost:29092 的代理，那么让我们尝试找出参与 Kafka 集群的所有活跃代理：

    ```bash
    $ kafka-broker-api-versions --bootstrap-server localhost:29092 | awk '/id/{print $1}'
    localhost:39092
    localhost:29092
    ```

    值得注意的是，我们使用 [awk](https://www.baeldung.com/linux/awk-guide) 命令过滤了输出，只显示了代理地址。此外，结果还正确显示了集群中有两个活动的代理。

    虽然这种方法看起来比 Zookeeper CLI 方法简单，但 kafka-broker-api-versions 二进制文件只是 Kafka 发行版最近才添加的。

5. shell 脚本

    在任何实际场景中，为每个代理手动执行 zkCli 或 zookeeper-shell 命令都会很累。因此，让我们编写一个 [Shell 脚本](https://www.baeldung.com/linux/linux-scripting-series)，将 Zookeeper 服务器地址作为输入，并返回给我们所有活动代理的列表。

    1. 辅助函数

        让我们在 functions.sh 脚本中写入所有辅助函数：

        ```bash
        $ cat functions.sh
        #!/bin/bash
        ZOOKEEPER_SERVER="${1:-localhost:2181}"
        # Helper Functions Below
        ```

        首先，让我们编写 get_broker_ids 函数，以获取将在内部调用 zkCli 命令的活动 broker id 集合：

        ```bash
        function get_broker_ids {
        broker_ids_out=$(zkCli -server $ZOOKEEPER_SERVER <<EOF
        ls /brokers/ids
        quit
        EOF
        )
        broker_ids_csv="$(echo "${broker_ids_out}" | grep '^\[.*\]$')"
        echo "$broker_ids_csv" | sed 's/\[//;s/]//;s/,/ /'
        }
        ```

        接下来，让我们编写 get_broker_details 函数，使用 broker_id 获取详细的经纪人信息：

        ```bash
        function get_broker_details {
        broker_id="$1"
        echo "$(zkCli -server $ZOOKEEPER_SERVER <<EOF
        get /brokers/ids/$broker_id
        quit
        EOF
        )"
        }
        ```

        现在我们有了详细的经纪人信息，让我们编写 parse_broker_endpoint 函数来获取经纪人的端点信息：

        ```sh
        function parse_endpoint_detail {
        broker_detail="$1"
        json="$(echo "$broker_detail"  | grep '^{.*}$')"
        json_endpoints="$(echo $json | jq .endpoints)"
        echo "$(echo $json_endpoints |jq . |  grep HOST | tr -d " ")"
        }
        ```

        在内部，我们使用 [jq](https://www.baeldung.com/linux/jq-command-json) 命令进行 JSON 解析。

    2. 主脚本

        现在，让我们编写使用 functions.sh 中定义的辅助函数的主脚本 get_all_active_brokers.sh：

        ```sh
        $ cat get_all_active_brokers.sh
        #!/bin/bash
        . functions.sh "$1"

        function get_all_active_brokers {
        broker_ids=$(get_broker_ids)
        for broker_id in $broker_ids
        do
            broker_details="$(get_broker_details $broker_id)"
            broker_endpoint=$(parse_endpoint_detail "$broker_details")
            echo "broker_id="$broker_id,"endpoint="$broker_endpoint
        done
        }

        get_all_active_brokers
        ```

        我们可以注意到，我们在 get_all_active_brokers 函数中遍历了所有 broker_id，以汇总所有活跃经纪人的端点。

        最后，让我们执行 get_all_active_brokers.sh 脚本，以便查看双节点 Kafka 集群的活动代理列表：

        ```sh
        $ ./get_all_active_brokers.sh localhost:2181
        broker_id=1,endpoint="PLAINTEXT_HOST://localhost:29092"
        broker_id=2,endpoint="PLAINTEXT_HOST://localhost:39092"
        ```

        我们可以看到结果是准确的。看来我们成功了！

6. 总结

    在本教程中，我们学习了如何使用 zookeeper-shell、zkCli 和 kafka-broker-api-versions 等 shell 命令来获取 Kafka 集群中的活动代理列表。此外，我们还编写了一个 shell 脚本，以便在实际场景中自动查找代理的详细信息。
