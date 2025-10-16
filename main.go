package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/joho/godotenv"
)

const (
	PORT           = "8081"
	OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
	ENDPOINT       = "/v1/chat/completions"

	// Feature flags
	ACCUMULATE_TEXT = false // Set to true to accumulate and show the complete text at the end

	// ANSI color codes for terminal output
	colorReset   = "\033[0m"
	colorRed     = "\033[31m"
	colorGreen   = "\033[32m"
	colorYellow  = "\033[33m"
	colorBlue    = "\033[34m"
	colorMagenta = "\033[35m"
	colorCyan    = "\033[36m"
	colorBold    = "\033[1m"
)

// Config holds the application configuration
type Config struct {
	OpenAIAPIKey string
}

// OpenAIChatCompletionRequest represents the structure of an OpenAI chat completion request
type OpenAIChatCompletionRequest struct {
	Model         string                   `json:"model"`
	Messages      []map[string]interface{} `json:"messages"`
	Stream        bool                     `json:"stream"`
	StreamOptions map[string]interface{}   `json:"stream_options,omitempty"`
}

// OpenAIUsage represents the token usage information
type OpenAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// OpenAIStreamResponse represents a chunk of OpenAI streaming response
type OpenAIStreamResponse struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
	} `json:"choices"`
}

// AnthropicStreamResponse represents a chunk of Anthropic streaming response
type AnthropicStreamResponse struct {
	Type  string `json:"type"`
	Delta struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"delta"`
}

// OpenAIResponsesStreamResponse represents a chunk of OpenAI Responses API streaming response
type OpenAIResponsesStreamResponse struct {
	Type  string `json:"type"`
	Delta string `json:"delta"`
}

// loadConfig loads the configuration from environment variables
func loadConfig() (*Config, error) {
	err := godotenv.Load()
	if err != nil {
		return nil, fmt.Errorf("error loading .env file: %v", err)
	}

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY is not set in the environment")
	}

	return &Config{
		OpenAIAPIKey: apiKey,
	}, nil
}

func main() {
	// Load configuration
	config, err := loadConfig()
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

	// Set up the HTTP server
	http.HandleFunc(ENDPOINT, createChatCompletionsHandler(config))

	fmt.Printf("Starting server on port %s...\n", PORT)
	log.Fatal(http.ListenAndServe(":"+PORT, nil))
}

// extractText extracts text content from different provider response formats
func extractText(data []byte) string {
	// Try to parse as OpenAI format
	var openAIResp OpenAIStreamResponse
	if err := json.Unmarshal(data, &openAIResp); err == nil {
		if len(openAIResp.Choices) > 0 && openAIResp.Choices[0].Delta.Content != "" {
			return openAIResp.Choices[0].Delta.Content
		}
	}

	// Try to parse as Anthropic format
	var anthropicResp AnthropicStreamResponse
	if err := json.Unmarshal(data, &anthropicResp); err == nil {
		if anthropicResp.Type == "content_block_delta" &&
			anthropicResp.Delta.Type == "text_delta" &&
			anthropicResp.Delta.Text != "" {
			return anthropicResp.Delta.Text
		}
	}

	// Try to parse as OpenAI Responses format
	var responsesResp OpenAIResponsesStreamResponse
	if err := json.Unmarshal(data, &responsesResp); err == nil {
		if responsesResp.Type == "response.output_text.delta" &&
			responsesResp.Delta != "" {
			return responsesResp.Delta
		}
	}

	// If none of the formats match or no text content is found
	return ""
}

// TimingInfo holds timing metrics for request processing
type TimingInfo struct {
	requestStart        time.Time
	requestPrepStart    time.Time
	requestPrepEnd      time.Time
	providerCallStart   time.Time
	providerResponseEnd time.Time
	firstChunkTime      time.Time
	completionTime      time.Time
}

// createChatCompletionsHandler creates a handler for the chat completions endpoint
func createChatCompletionsHandler(config *Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Initialize timing info
		timing := TimingInfo{
			requestStart: time.Now(),
		}

		// Only allow POST requests
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Create a new request to OpenAI
		timing.requestPrepStart = time.Now()
		forwardedReq, err := createOpenAIRequest(r, config)
		if err != nil {
			http.Error(w, fmt.Sprintf("Error creating request: %v", err), http.StatusInternalServerError)
			return
		}
		timing.requestPrepEnd = time.Now()

		// Send the request to OpenAI
		timing.providerCallStart = time.Now()
		client := &http.Client{}
		resp, err := client.Do(forwardedReq)
		if err != nil {
			http.Error(w, fmt.Sprintf("Error sending request to OpenAI: %v", err), http.StatusInternalServerError)
			return
		}
		timing.providerResponseEnd = time.Now()
		defer resp.Body.Close()

		// Set appropriate headers for streaming response
		for k, v := range resp.Header {
			for _, val := range v {
				w.Header().Add(k, val)
			}
		}
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("Transfer-Encoding", "chunked")
		w.WriteHeader(resp.StatusCode)

		// Handle streaming response
		if isStreamingRequest(r) {
			handleStreamingResponse(w, resp.Body, &timing)
		} else {
			// For non-streaming responses
			body, err := io.ReadAll(resp.Body)
			if err != nil {
				http.Error(w, fmt.Sprintf("Error reading response from OpenAI: %v", err), http.StatusInternalServerError)
				return
			}

			// Log the response summary
			fmt.Printf("%s%sResponse from provider:%s\n", colorBold, colorBlue, colorReset)

			// Write response back to client
			w.Write(body)

			// Record completion time
			timing.completionTime = time.Now()

			// Extract and log usage information if present
			extractAndLogUsage(body)

			// Print timing summary
			printTimingSummary(&timing, false)
		}
	}
}

// createOpenAIRequest creates a new request to be sent to OpenAI
func createOpenAIRequest(r *http.Request, config *Config) (*http.Request, error) {
	// Read the request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, fmt.Errorf("error reading request body: %v", err)
	}
	r.Body.Close()
	r.Body = io.NopCloser(bytes.NewBuffer(body))

	// Create a new request to OpenAI
	req, err := http.NewRequest(r.Method, OPENAI_API_URL, bytes.NewBuffer(body))
	if err != nil {
		return nil, fmt.Errorf("error creating request: %v", err)
	}

	// Copy headers from original request to new request, skipping "Posit" headers
	for key, values := range r.Header {
		if !strings.HasPrefix(key, "Posit") {
			for _, value := range values {
				req.Header.Add(key, value)
			}
		}
	}

	// Set content type and authorization headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+config.OpenAIAPIKey)

	return req, nil
}

// isStreamingRequest checks if the request is a streaming request
func isStreamingRequest(r *http.Request) bool {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return false
	}
	defer r.Body.Close()
	r.Body = io.NopCloser(bytes.NewBuffer(body))

	var req OpenAIChatCompletionRequest
	err = json.Unmarshal(body, &req)
	if err != nil {
		return false
	}

	return req.Stream
}

// handleStreamingResponse handles the streaming response from LLM providers
func handleStreamingResponse(w http.ResponseWriter, responseBody io.ReadCloser, timing *TimingInfo) {
	scanner := bufio.NewScanner(responseBody)
	var accumulatedText []string
	var usageLine []byte // Store usage data for display at the end

	// Flush header immediately
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}

	for scanner.Scan() {
		// Record time of first chunk if we haven't yet
		if timing.firstChunkTime.IsZero() {
			timing.firstChunkTime = time.Now()
		}

		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		// Process the line

		// Check if the line starts with "data: "
		if bytes.HasPrefix(line, []byte("data: ")) {
			dataStr := line[6:] // Remove 'data: ' prefix

			// Try to parse the JSON
			var data interface{}
			if err := json.Unmarshal(dataStr, &data); err == nil && ACCUMULATE_TEXT {
				// Extract text using provider-specific extraction
				text := extractText(dataStr)
				if text != "" {
					accumulatedText = append(accumulatedText, text)
				}
			}
		}

		// Save the line if it contains usage information for later display
		if bytes.HasPrefix(line, []byte("data: ")) && bytes.Contains(line, []byte("usage")) {
			// Store the usage data line for later processing
			usageLine = make([]byte, len(line))
			copy(usageLine, line)
		}

		// Write the line to the response with colorized output for debugging
		if bytes.HasPrefix(line, []byte("data: ")) {
			// Send the original data line to the client
			w.Write(line)
			w.Write([]byte("\n"))

			// Print colorized debug info to the console with pretty formatting
			dataStr := line[6:] // Remove 'data: ' prefix

			// Pretty print the JSON while maintaining field order
			prettyJSON := prettyPrintJSON(string(dataStr))
			fmt.Printf("%sdata: %s%s\n\n", colorCyan, prettyJSON, colorReset)
		} else {
			// For non-data lines (like empty lines that are part of SSE protocol)
			w.Write(line)
			w.Write([]byte("\n"))
		}

		// Flush to send the chunk immediately for responsive streaming
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
	}

	// Record completion time
	timing.completionTime = time.Now()

	// Extract and log usage information if available
	if len(usageLine) > 0 {
		extractAndLogUsage(usageLine)
	}

	// Print timing summary
	printTimingSummary(timing, true)

	// Log accumulated text if enabled (after timing summary)
	if ACCUMULATE_TEXT && len(accumulatedText) > 0 {
		fmt.Printf("\n%s%s===========================================================%s\n", colorBold, colorYellow, colorReset)
		fmt.Printf("%s%sAccumulated Text Response%s\n", colorBold, colorYellow, colorReset)
		fmt.Printf("%s%s===========================================================%s\n", colorBold, colorYellow, colorReset)
		fmt.Println(strings.Join(accumulatedText, ""))
		fmt.Printf("%s===========================================================%s\n", colorYellow, colorReset)
	}
}

// printTimingSummary prints a detailed summary of all timing metrics
func printTimingSummary(timing *TimingInfo, isStreaming bool) {
	requestType := "non-streaming"
	if isStreaming {
		requestType = "streaming"
	}

	fmt.Printf("\n%s%s======== TIMING SUMMARY (%s request) ========%s\n", colorBold, colorBlue, requestType, colorReset)

	// Request preparation time
	requestPrepTime := timing.requestPrepEnd.Sub(timing.requestPrepStart)
	fmt.Printf("%s• Request preparation: %v%s\n", colorGreen, requestPrepTime, colorReset)
	fmt.Printf("  ↳ %sStarts:%s Creating the request to forward to provider\n", colorCyan, colorReset)
	fmt.Printf("  ↳ %sEnds:%s Request fully prepared with headers and body\n", colorCyan, colorReset)

	// Provider response time
	providerRespTime := timing.providerResponseEnd.Sub(timing.providerCallStart)
	fmt.Printf("%s• Provider initial response: %v%s\n", colorGreen, providerRespTime, colorReset)
	fmt.Printf("  ↳ %sStarts:%s Sending request to provider API\n", colorCyan, colorReset)
	fmt.Printf("  ↳ %sEnds:%s First response headers received from provider\n", colorCyan, colorReset)

	// First chunk latency (streaming only)
	if isStreaming && !timing.firstChunkTime.IsZero() {
		firstChunkLatency := timing.firstChunkTime.Sub(timing.requestStart)
		fmt.Printf("%s• First content chunk latency: %v%s\n", colorGreen, firstChunkLatency, colorReset)
		fmt.Printf("  ↳ %sStarts:%s Initial request received by proxy\n", colorCyan, colorReset)
		fmt.Printf("  ↳ %sEnds:%s First content chunk received from provider\n", colorCyan, colorReset)
	}

	// Total request time
	totalRequestTime := timing.completionTime.Sub(timing.requestStart)
	fmt.Printf("%s• Total request time: %v%s\n", colorBold, totalRequestTime, colorReset)
	fmt.Printf("  ↳ %sStarts:%s Initial request received by proxy\n", colorCyan, colorReset)
	fmt.Printf("  ↳ %sEnds:%s Complete response finished processing\n", colorCyan, colorReset)

	fmt.Printf("%s%s=================================================%s\n\n", colorBold, colorBlue, colorReset)
}

// prettyPrintJSON formats a JSON string with indentation while preserving field order
func prettyPrintJSON(jsonStr string) string {
	// Check if it's valid JSON first
	var tmp interface{}
	if err := json.Unmarshal([]byte(jsonStr), &tmp); err != nil {
		return jsonStr // Not valid JSON, return as is
	}

	// Regex-based pretty printing to maintain field order
	level := 0
	inQuote := false
	inEscape := false
	prettyStr := strings.Builder{}

	// Add newline after these characters when not in a string
	addNewlineAfter := map[byte]bool{
		'{': true,
		'[': true,
		',': true,
	}

	// Add newline before these characters when not in a string
	addNewlineBefore := map[byte]bool{
		'}': true,
		']': true,
	}

	// Process each character
	for i := 0; i < len(jsonStr); i++ {
		ch := jsonStr[i]

		// Handle string quotes and escaping
		if ch == '"' && !inEscape {
			inQuote = !inQuote
		}
		inEscape = (ch == '\\' && !inEscape && inQuote)

		// Add character to output
		prettyStr.WriteByte(ch)

		// If we're not inside a quoted string
		if !inQuote {
			// Handle indentation after certain characters
			if addNewlineAfter[ch] {
				prettyStr.WriteString("\n")
				if ch == '{' || ch == '[' {
					level++
				}
				// Add indentation
				for j := 0; j < level; j++ {
					prettyStr.WriteString("  ")
				}
			} else if addNewlineBefore[ch] && i > 0 {
				// Remove any existing spaces before the closing bracket
				// First, trim any trailing whitespace
				currStr := prettyStr.String()
				currStr = strings.TrimRight(currStr[:len(currStr)-1], " \t\n")

				// Reset the builder and add the trimmed content back
				prettyStr = strings.Builder{}
				prettyStr.WriteString(currStr)

				// Add newline and proper indentation
				level--
				prettyStr.WriteString("\n")
				for j := 0; j < level; j++ {
					prettyStr.WriteString("  ")
				}
				prettyStr.WriteByte(ch)
			}
		}

		// Add space after colon if not in a string
		if ch == ':' && !inQuote {
			prettyStr.WriteString(" ")
		}
	}

	return prettyStr.String()
}

// extractAndLogUsage extracts and logs the token usage information
func extractAndLogUsage(data []byte) {
	// Check if we need to extract from "data:" prefix
	var jsonData []byte
	if bytes.HasPrefix(data, []byte("data: ")) {
		jsonData = data[6:] // Remove 'data: ' prefix
	} else {
		jsonData = data
	}

	// Try to parse the JSON
	var responseObj map[string]interface{}
	if err := json.Unmarshal(jsonData, &responseObj); err != nil {
		fmt.Printf("%sFailed to parse usage info: %v%s\n", colorRed, err, colorReset)
		return
	}

	// Variables to store usage information
	var promptTokens, completionTokens, totalTokens int
	var foundUsage bool

	// Check if the response has usage information directly
	if usage, ok := responseObj["usage"].(map[string]interface{}); ok {
		// Handle OpenAI format
		if pt, ok := usage["prompt_tokens"].(float64); ok {
			promptTokens = int(pt)
			completionTokens = int(usage["completion_tokens"].(float64))
			totalTokens = int(usage["total_tokens"].(float64))
			foundUsage = true
		}
	}

	// If not found in standard format, try alternative formats
	if !foundUsage {
		// For OpenAI streaming responses, usage might be in a different format
		if choices, ok := responseObj["choices"].([]interface{}); ok && len(choices) > 0 {
			choice := choices[0].(map[string]interface{})
			if finish_reason, ok := choice["finish_reason"].(string); ok && finish_reason != "" {
				if usage, ok := responseObj["usage"].(map[string]interface{}); ok {
					if pt, ok := usage["prompt_tokens"].(float64); ok {
						promptTokens = int(pt)
						completionTokens = int(usage["completion_tokens"].(float64))
						totalTokens = int(usage["total_tokens"].(float64))
						foundUsage = true
					}
				}
			}
		}
	}

	// Print usage summary if found
	if foundUsage {
		fmt.Printf("\n%s%s============= TOKEN USAGE SUMMARY ==============%s\n", colorBold, colorGreen, colorReset)
		fmt.Printf("%s• Input tokens:  %d%s\n", colorGreen, promptTokens, colorReset)
		fmt.Printf("%s• Output tokens: %d%s\n", colorGreen, completionTokens, colorReset)
		fmt.Printf("%s• Total tokens:  %d%s\n", colorGreen, totalTokens, colorReset)
		fmt.Printf("%s%s=================================================%s\n", colorBold, colorGreen, colorReset)
	}
}
