package util

import (
	"bytes"
	"fmt"
	"log"
	"os"
	"os/exec"
	"syscall"
)

func printError(err error, app string) {
	if err != nil {
		_, _ = os.Stderr.WriteString(fmt.Sprintf("["+app+"] Error: %s\n", err.Error()))
	}
}

func Execute(app string, args []string, dir string) string {
	log.Printf("Executing %v %v\n", app, args)
	cmd := exec.Command(app, args...)
	if dir != "" {
		cmd.Dir = dir
	}
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	outputBytes := &bytes.Buffer{}
	cmd.Stdout = outputBytes

	err := cmd.Run()
	printError(err, app)

	syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)

	return string(outputBytes.Bytes())
}
