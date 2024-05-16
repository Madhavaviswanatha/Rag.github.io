package com.example.ragchatbot.controller;

import io.milvus.client.MilvusClient;
import io.milvus.param.collection.*;
import io.milvus.param.dml.InsertParam;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/upload")
public class UploadController {

    @Autowired
    private MilvusClient milvusClient;

    @PostMapping("/file")
    public String uploadFile(@RequestParam("file") MultipartFile file) throws IOException {
        String text = new String(file.getBytes(), StandardCharsets.UTF_8);
        List<String> chunks = chunkText(text);
        List<float[]> embeddings = generateEmbeddings(chunks);
        storeEmbeddings(chunks, embeddings);
        return "File uploaded and processed.";
    }

    @PostMapping("/text")
    public String uploadText(@RequestParam("text") String text) {
        List<String> chunks = chunkText(text);
        List<float[]> embeddings = generateEmbeddings(chunks);
        storeEmbeddings(chunks, embeddings);
        return "Text uploaded and processed.";
    }

    private List<String> chunkText(String text) {
        // Simple chunking logic
        int chunkSize = 512;
        return text.lines().collect(Collectors.groupingBy(s -> s.hashCode() / chunkSize))
                .values().stream().map(lines -> String.join(" ", lines)).collect(Collectors.toList());
    }

    private List<float[]> generateEmbeddings(List<String> chunks) {
        // Mock embedding generation
        return chunks.stream().map(chunk -> new float[768]).collect(Collectors.toList());
    }

    private void storeEmbeddings(List<String> chunks, List<float[]> embeddings) {
        // Store embeddings in Milvus
        String collectionName = "chatbot_collection";
        if (!milvusClient.hasCollection(HasCollectionParam.newBuilder()
                .withCollectionName(collectionName).build()).getData()) {
            milvusClient.createCollection(CreateCollectionParam.newBuilder()
                    .withCollectionName(collectionName)
                    .withDescription("Chatbot Embeddings")
                    .withFields(List.of(
                            FieldType.newBuilder()
                                    .withName("embedding")
                                    .withDataType(DataType.FLOAT_VECTOR)
                                    .withDimension(768)
                                    .build(),
                            FieldType.newBuilder()
                                    .withName("chunk")
                                    .withDataType(DataType.VARCHAR)
                                    .withMaxLength(2048)
                                    .build()
                    )).build());
        }

        List<InsertParam.Field> fields = List.of(
                new InsertParam.Field("embedding", embeddings),
                new InsertParam.Field("chunk", chunks)
        );

        milvusClient.insert(InsertParam.newBuilder()
                .withCollectionName(collectionName)
                .withFields(fields)
                .build());
    }
}
