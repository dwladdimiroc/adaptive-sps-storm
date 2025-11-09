storm kill -w 0 exampleApp
storm jar exampleApp.jar com.github.dwladdimiroc.exampleApp.topology.ExampleTopology $1 $2
