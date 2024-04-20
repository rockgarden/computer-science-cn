# 使用Minikube的本地Docker镜像

1. 概述

    在本教程中，我们将把Docker容器部署到Kubernetes，并看看我们如何为这些容器使用本地镜像。我们将使用Minikube来运行Kubernetes集群。

2. Docker文件

    首先，我们需要一个[Dockerfile](https://docs.docker.com/engine/reference/builder/)，以便能够创建本地Docker镜像。这应该很简单，因为我们将专注于Minikube的命令。

    让我们创建一个Dockerfile，只用一个echo命令来打印一条信息：

    ```Dockerfile
    FROM alpine 
    CMD ["echo", "Hello World"]
    ```

3. docker-env命令

    对于第一个方法，我们需要确保[Docker CLI](https://docs.docker.com/engine/reference/commandline/cli/)已经安装。这是一个用于管理Docker资源的工具，比如镜像和容器。

    默认情况下，它使用我们机器上的Docker引擎，但我们可以很容易地改变它。我们将使用这个，并将我们的Docker CLI指向Minikube内部的Docker引擎。

    让我们检查一下这个先决条件，看看Docker CLI是否在工作：

    `$ docker version`

    让我们继续接下来的步骤。我们可以配置这个CLI来使用Minikube内部的Docker引擎。这样，我们就能列出Minikube中可用的图像，甚至在里面构建图像。

    让我们看看配置Docker CLI所需的[步骤](https://minikube.sigs.k8s.io/docs/commands/docker-env/)：

    `$ minikube docker-env`

    我们可以看到这里的命令列表：

    ```log
    export DOCKER_TLS_VERIFY="1"
    export DOCKER_HOST="tcp://172.22.238.61:2376"
    export DOCKER_CERT_PATH="C:\Users\Baeldung\.minikube\certs"
    export MINIKUBE_ACTIVE_DOCKERD="minikube"

    # To point your shell to minikube's docker-daemon, run:
    # eval $(minikube -p minikube docker-env)
    ```

    让我们执行最后一行的命令，因为它将为我们做配置：

    `$ eval $(minikube -p minikube docker-env)`

    现在，我们可以使用Docker CLI来调查Minikube内部的Docker环境。

    让我们用minikube image ls命令列出可用的图像：

    `$ minikube image ls --format table`

    如果我们将其与[docker image ls](https://docs.docker.com/engine/reference/commandline/image_ls/)命令的输出进行比较，我们会发现两者都显示相同的列表。这意味着我们的Docker CLI已经正确配置了。

    让我们使用我们的Docker文件并从中构建一个镜像：

    `$ docker build -t first-image -f ./Dockerfile .`

    现在它在Minikube中是可用的，我们可以创建一个使用这个镜像的pod：

    `$ kubectl run first-container --image=first-image --image-pull-policy=Never --restart=Never`

    让我们检查一下这个pod的日志：

    `$ kubectl logs first-container`

    我们可以看到预期的 "Hello World" 信息。一切工作正常。让我们关闭终端，以确保我们的Docker CLI在下一个例子中没有连接到Minikube。

4. Minikube图像加载命令

    让我们看看另一种使用本地镜像的方法。这一次，我们将在我们的机器上建立Minikube之外的Docker镜像，并将其加载到Minikube。让我们来构建这个镜像：

    `$ docker build -t second-image -f ./Dockerfile .`

    现在这个镜像已经存在，但它在Minikube中还不可用。让我们来加载它：

    `$ minikube image load second-image`

    让我们列出图像并检查它是否可用：

    `$ minikube image ls --format table`

    我们可以在列表中看到新的图像。这意味着我们可以创建pod：

    `$ kubectl run second-container --image=second-image --image-pull-policy=Never --restart=Never`

    容器成功启动。让我们检查一下日志：

    `$ kubectl logs second-container`

    我们可以看到它打印了正确的信息。

5. Minikube镜像构建命令

    在前面的例子中，我们加载了一个预构建的Docker镜像到Minikube。然而，我们也可以在Minikube内部构建我们的图像。

    让我们使用相同的Docker文件，构建一个新的Docker镜像：

    `$ minikube image build -t third-image -f ./Dockerfile .`

    现在这个镜像在Minikube中是可用的，我们可以用它启动一个容器：

    `$ kubectl run third-container --image=third-image --image-pull-policy=Never --restart=Never`

    让我们检查一下日志，确保它在工作：

    `$ kubectl logs third-container`

    它如期打印出了 "Hello World" 信息。

6. 总结

    在这篇文章中，我们用三种不同的方式在Minikube中运行本地Docker镜像。

    首先，我们配置了我们的Docker CLI以连接到Minikube内部的Docker引擎。然后，我们看到了两个命令来加载一个预构建的图像和直接在Minikube中构建一个图像。

## 相关文章

- [ ] [Using Local Docker Images With Minikube](https://www.baeldung.com/docker-local-images-minikube)
