package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"

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
	dict     map[string][]string
	cilinMap map[string]CiLinCode // 词语到编码的映射
	codemap  map[string][]string  // 编码到词语的映射
	mu       sync.RWMutex
}

// 全局同义词字典
var globalSynonymDict = &SynonymDict{
	dict:     make(map[string][]string),
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

		// 将同一编码下的词语互相设为同义词
		for i, word := range words {
			if sd.dict[word] == nil {
				sd.dict[word] = make([]string, 0)
			}
			for j, synonym := range words {
				if i != j && !contains(sd.dict[word], synonym) {
					sd.dict[word] = append(sd.dict[word], synonym)
				}
			}
		}
		sd.mu.Unlock()
	}

	return scanner.Err()
}

// DownloadCiLinDict 下载哈工大同义词词林
func (sd *SynonymDict) DownloadCiLinDict() error {
	// 使用本地默认同义词表
	defaultSynonyms := map[string][]string{
		"非常": {"很", "特别", "极其", "十分", "格外"},
		"好吃": {"美味", "可口", "美味可口", "美味佳肴", "佳肴", "美食"},
		"喜欢": {"爱", "热爱", "钟爱", "喜爱", "爱好"},
		"漂亮": {"美丽", "好看", "美观", "动人", "标致"},
		"快乐": {"开心", "高兴", "愉快", "欢乐", "欣喜"},
		"生气": {"愤怒", "恼怒", "发火", "动怒", "光火"},
		"聪明": {"智慧", "伶俐", "智慧", "明智", "睿智"},
		"努力": {"奋斗", "拼搏", "用功", "勤奋", "奋进"},
		"成功": {"胜利", "成就", "达成", "实现", "完成"},
		"重要": {"关键", "主要", "核心", "关键", "要害"},
		"应该": {"必须", "需要", "理应", "应当", "须要"},
		"保护": {"爱护", "维护", "呵护", "保卫", "守护"},
		"独特": {"特别", "特殊", "与众不同", "别致", "新颖"},
		"动听": {"悦耳", "好听", "优美", "美妙", "动人"},
		"复杂": {"繁杂", "繁复", "纷繁", "错综", "难解"},
		"今天": {"今日", "这天", "当天", "这一天"},
		"工作": {"劳动", "事业", "职业", "事务", "任务"},
		"认真": {"严谨", "专注", "细致", "用心", "专心"},
		"味道": {"口味", "滋味", "风味", "口感", "味儿"},
		"问题": {"疑问", "难题", "困难", "课题", "难点"},
	}

	sd.mu.Lock()
	defer sd.mu.Unlock()

	// 将默认同义词添加到词典中
	for word, synonyms := range defaultSynonyms {
		if sd.dict[word] == nil {
			sd.dict[word] = make([]string, 0)
		}
		for _, synonym := range synonyms {
			if !contains(sd.dict[word], synonym) {
				sd.dict[word] = append(sd.dict[word], synonym)
			}
			// 反向添加
			if sd.dict[synonym] == nil {
				sd.dict[synonym] = make([]string, 0)
			}
			if !contains(sd.dict[synonym], word) {
				sd.dict[synonym] = append(sd.dict[synonym], word)
			}
		}
	}

	return nil
}

// GetSynonymsByCiLin 通过哈工大词林获取同义词
func (sd *SynonymDict) GetSynonymsByCiLin(word string) []string {
	sd.mu.RLock()
	defer sd.mu.RUnlock()

	// 获取词的编码
	code, ok := sd.cilinMap[word]
	if !ok {
		return nil
	}

	// 收集所有相同编码的词
	var synonyms []string
	for w, c := range sd.cilinMap {
		if w != word && // 不包含自己
			c.FirstLevel == code.FirstLevel && // 第一层相同
			c.SecondLevel == code.SecondLevel && // 第二层相同
			c.ThirdLevel == code.ThirdLevel { // 第三层相同
			synonyms = append(synonyms, w)
		}
	}

	return synonyms
}

// GetSynonyms 获取同义词（结合多个来源）
func (sd *SynonymDict) GetSynonyms(word string) []string {
	// 首先获取词林中的同义词
	cilinSynonyms := sd.GetSynonymsByCiLin(word)

	sd.mu.RLock()
	// 获取基础词典中的同义词
	basicSynonyms := sd.dict[word]
	sd.mu.RUnlock()

	// 合并两个来源的同义词
	result := make([]string, 0)
	result = append(result, cilinSynonyms...)

	// 添加基础词典中的同义词（去重）
	for _, syn := range basicSynonyms {
		if !contains(result, syn) {
			result = append(result, syn)
		}
	}

	return result
}

// contains 检查切片是否包含某个字符串
func contains(slice []string, str string) bool {
	for _, s := range slice {
		if s == str {
			return true
		}
	}
	return false
}

// LoadSynonymDict 加载同义词字典
func (sd *SynonymDict) LoadSynonymDict(filePath string) error {
	file, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		words := strings.Split(line, " ")
		if len(words) < 2 {
			continue
		}

		// 将行内所有词互相设置为同义词
		for i, word := range words {
			for j, synonym := range words {
				if i != j {
					sd.mu.Lock()
					if sd.dict[word] == nil {
						sd.dict[word] = make([]string, 0)
					}
					// 检查是否已存在
					exists := false
					for _, w := range sd.dict[word] {
						if w == synonym {
							exists = true
							break
						}
					}
					if !exists {
						sd.dict[word] = append(sd.dict[word], synonym)
					}
					sd.mu.Unlock()
				}
			}
		}
	}

	return scanner.Err()
}

// DownloadSynonymDict 下载同义词典
func (sd *SynonymDict) DownloadSynonymDict() error {
	// 可以从多个来源下载
	sources := []string{
		"https://raw.githubusercontent.com/fighting41love/funNLP/master/data/同义词库.txt",
		// 添加其他来源
	}

	for _, url := range sources {
		resp, err := http.Get(url)
		if err != nil {
			continue
		}
		defer resp.Body.Close()

		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			continue
		}

		// 创建临时文件
		tmpFile, err := os.CreateTemp("", "synonym_dict_*.txt")
		if err != nil {
			continue
		}
		defer os.Remove(tmpFile.Name())

		if _, err := tmpFile.Write(body); err != nil {
			continue
		}

		// 加载下载的词典
		if err := sd.LoadSynonymDict(tmpFile.Name()); err != nil {
			continue
		}
	}

	return nil
}

// SaveSynonymDict 保存同义词典到文件
func (sd *SynonymDict) SaveSynonymDict(filePath string) error {
	sd.mu.RLock()
	defer sd.mu.RUnlock()

	// 确保目录存在
	dir := filepath.Dir(filePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	file, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	return encoder.Encode(sd.dict)
}

// LoadSynonymDictFromJSON 从JSON文件加载同义词典
func (sd *SynonymDict) LoadSynonymDictFromJSON(filePath string) error {
	file, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	sd.mu.Lock()
	defer sd.mu.Unlock()

	decoder := json.NewDecoder(file)
	return decoder.Decode(&sd.dict)
}

// 初始化同义词典
func initSynonymDict() error {
	// 创建数据目录
	dataDir := "data"
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return err
	}

	dictPath := filepath.Join(dataDir, "synonym_dict.json")

	// 首先尝试加载本地词典
	if err := globalSynonymDict.LoadSynonymDictFromJSON(dictPath); err != nil {
		log.Println("本地词典加载失败，尝试下载...")

		// 下载基础同义词典
		if err := globalSynonymDict.DownloadSynonymDict(); err != nil {
			log.Printf("基础同义词典下载失败: %v", err)
		}

		// 下载哈工大词林
		if err := globalSynonymDict.DownloadCiLinDict(); err != nil {
			log.Printf("哈工大词林下载失败: %v", err)
		}

		// 保存到本地以供将来使用
		if err := globalSynonymDict.SaveSynonymDict(dictPath); err != nil {
			log.Printf("保存词典失败: %v", err)
		}
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

	// 保存为新的Excel文件
	newFilePath := "output.xlsx"
	if err := outFile.SaveAs(newFilePath); err != nil {
		log.Fatalf("保存结果文件失败：%v", err)
	}

	fmt.Printf("计算完成，结果已保存到当前目录下的: %s\n", newFilePath)
}
