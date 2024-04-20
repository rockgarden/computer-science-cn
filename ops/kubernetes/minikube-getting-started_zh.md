# 开始使用Minikube

1. 概述

    Kubernetes是一个流行的开源平台，用于管理容器化工作负载和服务。它提供了一种方法来管理和协调容器化的应用程序，根据需要扩大和缩小它们，并确保它们可靠和安全地运行。

    [Minikube](https://minikube.sigs.k8s.io/docs/)是一个工具，使我们能够在我们的机器上运行一个本地的、单节点的Kubernetes集群，这是测试和开发的理想选择。在本教程中，我们将介绍在本地机器上安装和使用Minikube的步骤。

2. 前提条件

    在我们开始之前，我们应该确保我们有以下条件：

    - 一台运行Linux、macOS或Windows的机器。
    - 一个管理程序，如VirtualBox或KVM，已经安装在我们的机器上。如果我们使用的是Windows，我们可以用Hyper-V代替。
    - 一个shell环境，如Bash或PowerShell。
    - 在我们的机器上安装了curl。
3. 安装Minikube

    第一步是在我们的机器上安装Minikube。我们可以按照Minikube网站上提供的说明来做： <https://minikube.sigs.k8s.io/docs/start/>

    或者，我们可以使用以下命令来安装最新版本的Minikube：

    `$ curl -Lo minikube https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64 && chmod +x minikube && sudo mv minikube /usr/local/bin/`

    这个命令是针对Linux的。如果我们使用的是macOS或Windows，我们将需要为我们的平台下载相应的二进制文件。

    一旦我们安装了Minikube，我们可以通过运行以下命令来验证它是否安装正确：

    `$ minikube version`

    该命令输出Minikube二进制文件的版本号。

4. 启动一个Minikube集群

    下一步是启动一个Minikube集群。我们可以通过运行下面的命令来做到这一点：

    `$ minikube start`

    这个命令将在我们的本地机器上启动一个虚拟机，并将其配置为运行一个单节点的Kubernetes集群。

    我们第一次运行这个命令时，它将为虚拟机下载必要的ISO镜像，这可能需要几分钟时间。一旦虚拟机运行，我们可以通过运行以下命令来检查集群的状态：

    `$ minikube status`

    这个输出显示，Minikube控制平面正在运行，kubelet和API服务器也在运行。kubeconfig已经配置好了，这意味着我们可以使用kubectl命令与集群互动。此外，输出还显示了Minikube主机的状态。

5. 与Minikube集群互动

    一旦集群启动并运行，我们可以使用kubectl命令行工具与之互动。kubectl是与Kubernetes集群互动的主要CLI工具，使我们能够部署应用程序，管理节点，并执行其他操作。

    为了在Minikube中使用kubectl，我们必须将上下文设置为Minikube集群。我们可以通过运行以下命令来做到这一点：

    `$ kubectl config use-context minikube`

    这个命令告诉kubectl使用Minikube集群的配置。

    我们现在可以运行kubectl命令来与Minikube集群进行交互。例如，我们可以运行下面的命令来获取集群中的节点信息：

    `$ kubectl get nodes`

    该命令显示了集群中的节点列表，以及它们的状态、角色、年龄和Kubernetes版本。

6. 部署一个应用程序

    为了将一个应用程序部署到Minikube集群，我们需要创建一个Kubernetes部署对象。部署对象管理一组应用程序的副本，并确保所需的状态得到维护。

    对于这个例子，让我们创建一个简单的部署对象，运行一个NGINX网络服务器。为此，我们首先创建一个名为nginx-deployment.yaml的文件，内容如下：

    ```yml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
    name: nginx-deployment
    spec:
    replicas: 1
    selector:
        matchLabels:
        app: nginx
    template:
        metadata:
        labels:
            app: nginx
        spec:
        containers:
        - name: nginx
            image: nginx
            ports:
            - containerPort: 80
    ```

    这个YAML文件定义了一个具有单个副本的部署对象，它运行一个NGINX容器。它为HTTP流量提供了80端口。

    然后，为了创建部署，我们运行以下命令：

    `$ kubectl apply -f nginx-deployment.yaml`

    这个命令在Minikube集群中创建了部署对象。

    为了访问NGINX网络服务器，我们需要把它作为一个Kubernetes[服务](https://www.baeldung.com/ops/kubernetes)公开。服务是一个抽象的概念，它将一组pod作为网络服务公开。让我们创建一个名为nginx-service.yaml的文件，内容如下：

    ```yml
    apiVersion: v1
    kind: Service
    metadata:
    name: nginx-service
    spec:
    type: NodePort
    selector:
        app: nginx
    ports:
    - name: http
        port: 80
        targetPort: 80
    ```

    这个YAML文件定义了一个服务对象，在虚拟机上随机分配的端口上公开NGINX部署。

    为了创建这个服务，我们运行以下命令：

    `$ kubectl apply -f nginx-service.yaml`

    现在我们可以通过获取虚拟机的IP地址和服务的端口号来访问NGINX网络服务器：

    `$ minikube service nginx-service --url`

    这应该会输出一个URL，我们可以用它来访问NGINX网络服务器。

    要在Minikube中使用[本地构建](https://www.baeldung.com/docker-local-images-minikube)的Docker镜像，我们可以简单地用Minikube Docker守护进程的IP地址来标记该镜像，并使用docker tag和docker push命令将其推送到Minikube Docker注册中心。

7. 总结

    在这篇文章中，我们介绍了安装和使用Minikube的步骤，以便在我们的机器上运行一个本地Kubernetes集群。我们还演示了如何将一个应用程序部署到集群中，并使用Kubernetes服务访问它。

    Minikube是一个用于测试和开发的伟大工具，它使我们能够快速、轻松地启动和运行Kubernetes。

## 使用Minikube的本地Docker镜像

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
