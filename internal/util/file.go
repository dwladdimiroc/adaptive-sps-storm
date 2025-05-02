package util

import (
	"bytes"
	"os"
	"regexp"
	"strings"
)

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func writeFile(data []byte, filename string) {
	folder := "stats/"
	errWrite := os.WriteFile(folder+filename, data, 0644)
	check(errWrite)
}

func ParseCPU(dataFile string, components string) {
	var buffer bytes.Buffer

	lines := strings.Split(dataFile, "\n")
	lines = lines[2 : len(lines)-1]

	for index, line := range lines {
		regular, errRegular := regexp.Compile(`\s+`)
		check(errRegular)
		line = regular.ReplaceAllString(line, " ")
		words := strings.Split(line, " ")

		for index, word := range words {
			if index != len(words)-1 {
				buffer.WriteString(word)
				buffer.WriteString(";")
			} else {
				buffer.WriteString(word)
			}
		}

		if index != len(lines)-1 {
			buffer.WriteString("\n")
		}
	}

	filename := "statsCPU-" + components + ".csv"
	writeFile(buffer.Bytes(), filename)

	buffer.Reset()

}

func ParseMemory(dataFile string, components string) {
	var buffer bytes.Buffer

	lines := strings.Split(dataFile, "\n")
	lines = lines[1 : len(lines)-1]

	for index, line := range lines {
		regular, errRegular := regexp.Compile(`\s+`)
		check(errRegular)
		line = regular.ReplaceAllString(line, " ")
		line = strings.TrimSpace(line)
		words := strings.Split(line, " ")

		for index, word := range words {
			if index != len(words)-1 {
				buffer.WriteString(word)
				buffer.WriteString(";")
			} else {
				buffer.WriteString(word)
			}
		}

		if index != len(lines)-1 {
			buffer.WriteString("\n")
		}
	}

	filename := "statsMem-" + components + ".csv"
	writeFile(buffer.Bytes(), filename)

	buffer.Reset()

}
