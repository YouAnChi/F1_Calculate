package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/go-ego/gse"
	"github.com/xuri/excelize/v2"
)

// CiLinCode 哈工大词林编码结构
type CiLinCode struct {
	FirstLevel  string // 第一层编码
	SecondLevel string // 第二层编码
	ThirdLevel  string // 第三层编码
	FourthLevel string // 第四层编码
	FifthLevel  string // 第五层编码
}

// SynonymDict 同义词字典
type SynonymDict struct {
	cilinMap map[string]CiLinCode // 词语到编码的映射
	codemap  map[string][]string  // 编码到词语的映射
	mu       sync.RWMutex
}

// 全局同义词字典
var globalSynonymDict = &SynonymDict{
	cilinMap: make(map[string]CiLinCode),
	codemap:  make(map[string][]string),
}

// ParseCiLinCode 解析哈工大词林编码
func ParseCiLinCode(code string) CiLinCode {
	return CiLinCode{
		FirstLevel:  code[0:1],
		SecondLevel: code[1:2],
		ThirdLevel:  code[2:4],
		FourthLevel: code[4:5],
		FifthLevel:  code[5:7],
	}
}

// LoadCiLinDict 加载哈工大同义词词林
func (sd *SynonymDict) LoadCiLinDict(filePath string) error {
	file, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, " ")
		if len(parts) < 2 {
			continue
		}

		// 第一部分是编码
		code := parts[0]
		cilinCode := ParseCiLinCode(code)
		words := parts[1:]

		// 更新编码到词语的映射
		sd.mu.Lock()
		sd.codemap[code] = words

		// 更新词语到编码的映射
		for _, word := range words {
			sd.cilinMap[word] = cilinCode
		}
		sd.mu.Unlock()
	}

	return scanner.Err()
}

// GetSynonyms 获取同义词
func (sd *SynonymDict) GetSynonyms(word string) []string {
	sd.mu.RLock()
	defer sd.mu.RUnlock()
	
	// 通过词林获取同义词
	if code, ok := sd.cilinMap[word]; ok {
		codeStr := fmt.Sprintf("%s%s%s%s%s", 
			code.FirstLevel, 
			code.SecondLevel, 
			code.ThirdLevel, 
			code.FourthLevel, 
			code.FifthLevel)
		if words, exists := sd.codemap[codeStr]; exists {
			synonyms := make([]string, 0)
			for _, w := range words {
				if w != word {
					synonyms = append(synonyms, w)
				}
			}
			return synonyms
		}
	}
	return nil
}

// initSynonymDict 初始化同义词典
func initSynonymDict() error {
	// 加载哈工大词林
	err := globalSynonymDict.LoadCiLinDict("data/cilin.txt")
	if err != nil {
		return fmt.Errorf("加载词林失败: %v", err)
	}
	return nil
}

// TextSimilarity 存储文本相似度的各种指标
type TextSimilarity struct {
	F1              float64
	Precision       float64
	Recall          float64
	SemanticF1      float64 // 考虑语义的F1值
	PositionAwareF1 float64 // 考虑位置的F1值
}

// WordMatch 存储词语匹配信息
type WordMatch struct {
	word     string
	score    float64
	position int
}

// calculateSemanticF1 计算考虑语义和位置信息的F1值
func calculateSemanticF1(actual, predicted string, seg gse.Segmenter) TextSimilarity {
	// 分词
	actualWords := seg.Cut(actual, true)
	predictedWords := seg.Cut(predicted, true)

	// 创建词频和位置映射
	actualMatches := make([]WordMatch, 0)
	predictedMatches := make([]WordMatch, 0)

	// 记录词语位置
	for i, word := range actualWords {
		actualMatches = append(actualMatches, WordMatch{
			word:     word,
			position: i,
			score:    1.0,
		})
	}

	for i, word := range predictedWords {
		predictedMatches = append(predictedMatches, WordMatch{
			word:     word,
			position: i,
			score:    1.0,
		})
	}

	// 计算匹配分数
	matches := calculateMatches(actualMatches, predictedMatches, len(actualWords), len(predictedWords))

	// 计算基础指标
	truePositives := matches.exactMatches
	semanticPositives := matches.semanticMatches
	totalActual := float64(len(actualWords))
	totalPredicted := float64(len(predictedWords))

	// 计算基础F1
	precision := safeDiv(float64(truePositives), totalPredicted)
	recall := safeDiv(float64(truePositives), totalActual)
	basicF1 := calculateF1Score(precision, recall)

	// 计算语义F1
	semanticPrecision := safeDiv(semanticPositives, totalPredicted)
	semanticRecall := safeDiv(semanticPositives, totalActual)
	semanticF1 := calculateF1Score(semanticPrecision, semanticRecall)

	// 计算位置感知F1
	positionPrecision := safeDiv(matches.positionAwareScore, totalPredicted)
	positionRecall := safeDiv(matches.positionAwareScore, totalActual)
	positionF1 := calculateF1Score(positionPrecision, positionRecall)

	return TextSimilarity{
		F1:              basicF1,
		Precision:       precision,
		Recall:          recall,
		SemanticF1:      semanticF1,
		PositionAwareF1: positionF1,
	}
}

type matchResult struct {
	exactMatches       float64
	semanticMatches    float64
	positionAwareScore float64
}

// calculateMatches 计算词语匹配情况
func calculateMatches(actual, predicted []WordMatch, actualLen, predictedLen int) matchResult {
	exactMatches := 0.0
	semanticMatches := 0.0
	positionAwareScore := 0.0

	// 创建访问标记
	usedPredicted := make([]bool, len(predicted))

	// 首先处理完全匹配
	for _, actualWord := range actual {
		for j, predictedWord := range predicted {
			if usedPredicted[j] {
				continue
			}

			matchScore := getMatchScore(actualWord.word, predictedWord.word)
			if matchScore > 0 {
				positionScore := calculatePositionScore(
					float64(actualWord.position)/float64(actualLen),
					float64(predictedWord.position)/float64(predictedLen),
				)

				if matchScore == 1.0 {
					exactMatches++
				}
				semanticMatches += matchScore
				positionAwareScore += matchScore * positionScore
				usedPredicted[j] = true
				break
			}
		}
	}

	return matchResult{
		exactMatches:       exactMatches,
		semanticMatches:    semanticMatches,
		positionAwareScore: positionAwareScore,
	}
}

// getMatchScore 获取两个词的匹配分数
func getMatchScore(word1, word2 string) float64 {
	// 完全匹配
	if word1 == word2 {
		return 1.0
	}

	// 检查同义词
	if synonyms := globalSynonymDict.GetSynonyms(word1); synonyms != nil {
		for _, syn := range synonyms {
			if syn == word2 {
				return 0.9
			}
		}
	}

	// 字符重叠度计算
	if len(word1) >= 2 && len(word2) >= 2 {
		overlap := calculateCharacterOverlap(word1, word2)
		if overlap > 0.5 {
			return overlap * 0.8
		}
	}

	return 0
}

// calculateCharacterOverlap 计算字符重叠度
func calculateCharacterOverlap(word1, word2 string) float64 {
	chars1 := strings.Split(word1, "")
	chars2 := strings.Split(word2, "")

	common := 0
	for _, c1 := range chars1 {
		for _, c2 := range chars2 {
			if c1 == c2 {
				common++
				break
			}
		}
	}

	return float64(common) / math.Max(float64(len(chars1)), float64(len(chars2)))
}

// calculatePositionScore 计算位置相似度分数
func calculatePositionScore(pos1, pos2 float64) float64 {
	diff := math.Abs(pos1 - pos2)
	return math.Exp(-3 * diff * diff) // 使用较温和的衰减率
}

func calculateF1Score(precision, recall float64) float64 {
	if precision+recall == 0 {
		return 0
	}
	return 2 * (precision * recall) / (precision + recall)
}

func safeDiv(a, b float64) float64 {
	if b == 0 {
		return 0
	}
	return a / b
}

func main() {
	// 初始化分词器
	var seg gse.Segmenter
	seg.LoadDict()

	// 初始化同义词典
	if err := initSynonymDict(); err != nil {
		log.Printf("警告：同义词典初始化失败：%v", err)
		log.Println("将使用基础同义词匹配")
	}

	// 获取用户输入的Excel路径
	var filePath string
	fmt.Print("请输入Excel文件路径: ")
	fmt.Scanln(&filePath)

	// 打开Excel文件
	f, err := excelize.OpenFile(filePath)
	if err != nil {
		log.Fatalf("打开Excel文件失败：%v", err)
	}
	defer f.Close()

	// 获取第一个工作表名称
	sheetName := f.GetSheetName(0)

	// 读取A列和B列数据
	rows, err := f.GetRows(sheetName)
	if err != nil {
		log.Fatalf("读取Excel数据失败：%v", err)
	}

	// 创建新的Excel文件
	outFile := excelize.NewFile()
	defer outFile.Close()

	// 添加表头
	headers := []string{"标准答案", "预测文本", "基础F1值", "语义F1值", "位置感知F1值", "精确率", "召回率"}
	for colIdx, header := range headers {
		cell, _ := excelize.CoordinatesToCellName(colIdx+1, 1)
		outFile.SetCellValue("Sheet1", cell, header)
	}

	// 计算相似度指标并写入Excel
	for rowIdx, row := range rows {
		if rowIdx == 0 || len(row) < 2 { // 跳过表头和不完整的行
			continue
		}

		// 获取A列和B列的中文文本
		actual := strings.TrimSpace(row[0])
		predicted := strings.TrimSpace(row[1])

		// 计算相似度指标
		similarity := calculateSemanticF1(actual, predicted, seg)

		// 写入各项指标
		rowNum := rowIdx + 1
		outFile.SetCellValue("Sheet1", fmt.Sprintf("A%d", rowNum), actual)
		outFile.SetCellValue("Sheet1", fmt.Sprintf("B%d", rowNum), predicted)
		outFile.SetCellValue("Sheet1", fmt.Sprintf("C%d", rowNum), similarity.F1)
		outFile.SetCellValue("Sheet1", fmt.Sprintf("D%d", rowNum), similarity.SemanticF1)
		outFile.SetCellValue("Sheet1", fmt.Sprintf("E%d", rowNum), similarity.PositionAwareF1)
		outFile.SetCellValue("Sheet1", fmt.Sprintf("F%d", rowNum), similarity.Precision)
		outFile.SetCellValue("Sheet1", fmt.Sprintf("G%d", rowNum), similarity.Recall)
	}

	// 调整列宽
	outFile.SetColWidth("Sheet1", "A", "G", 20)

	// 生成带时间戳的文件名
	timestamp := time.Now().Format("2006-01-02_15-04-05")
	newFilePath := fmt.Sprintf("output_%s.xlsx", timestamp)
	if err := outFile.SaveAs(newFilePath); err != nil {
		log.Fatalf("保存结果文件失败：%v", err)
	}

	fmt.Printf("计算完成，结果已保存到当前目录下的: %s\n", newFilePath)
}
