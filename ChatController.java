package com.example.ragchatbot.controller;

import io.milvus.client.MilvusClient;
import io.milvus.param.dml.SearchParam;
import io.milvus.param.index.KnnSearchParam;
import io.milvus.param.dml.SearchResultsWrapper;
import com.theokanning.openai.OpenAiService;
import com.theokanning.openai.completion.CompletionRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/chat")
public class ChatController {

    @Autowired
    private MilvusClient milvusClient;

    @Autowired
    private OpenAiService openAiService;

    @PostMapping("/query")
    public String chat(@RequestParam("query") String query) {
        float[] queryEmbedding = generateEmbedding(query);
        List<String> contextChunks = retrieveRelevantChunks(queryEmbedding);
        String prompt = buildPrompt(query, contextChunks);
        return generateResponse(prompt);
    }

    private float[] generateEmbedding(String text) {
        // Mock embedding generation
        return new float[768];
    }

    private List<String> retrieveRelevantChunks(float[] embedding) {
        String collectionName = "chatbot_collection";
        SearchParam searchParam = SearchParam.newBuilder()
                .withCollectionName(collectionName)
                .withMetricType(KnnSearchParam.MetricType.L2)
                .withTopK(5)
                .withVectors(Collections.singletonList(embedding))
                .build();
        SearchResultsWrapper results = milvusClient.search(searchParam);
        return results.getFieldData("chunk").stream().map(Object::toString).collect(Collectors.toList());
    }

    private String buildPrompt(String query, List<String> contextChunks) {
        return String.join("\n", contextChunks) + "\n\nUser: " + query + "\nBot:";
    }

    private String generateResponse(String prompt) {
        CompletionRequest completionRequest = CompletionRequest.builder()
                .prompt(prompt)
                .maxTokens(150)
                .temperature(0.9)
                .build();
        return openAiService.createCompletion("text-davinci-003", completionRequest).getChoices().get(0).getText();
    }
}
