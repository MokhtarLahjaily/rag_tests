package ma.emsi.lahjaily;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.net.URI;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class TestRoutage {

    // Helper pour récupérer le chemin du fichier de ressources
    private static Path getPath(String fileName) {
        try {
            URI fileUri = TestRoutage.class.getClassLoader().getResource(fileName).toURI();
            return Paths.get(fileUri);
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }

    // Fonction de configuration du Logger
    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    /**
     * Méthode helper pour charger, parser, splitter et stocker un document
     * dans un EmbeddingStore en mémoire.
     */
    private static EmbeddingStore<TextSegment> ingestDocument(String resourceName, EmbeddingModel embeddingModel) {
        Path documentPath = getPath(resourceName);
        DocumentParser parser = new ApacheTikaDocumentParser();
        Document document = FileSystemDocumentLoader.loadDocument(documentPath, parser);
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        Response<List<Embedding>> embeddingsResponse = embeddingModel.embedAll(segments);
        embeddingStore.addAll(embeddingsResponse.content(), segments);
        System.out.println("Ingestion de '" + resourceName + "' terminée. " + segments.size() + " segments.");
        return embeddingStore;
    }

    public static void main(String[] args) {
        configureLogger();
        String llmKey = System.getenv("GEMINI_KEY");

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(llmKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequests(true)
                .logResponses(true)
                .build();

        // Modèle d'embedding partagé
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // === PHASE 1: INGESTION (2 documents) ===
        EmbeddingStore<TextSegment> ragStore = ingestDocument("rag.pdf", embeddingModel);
        EmbeddingStore<TextSegment> financeStore = ingestDocument("finance.pdf", embeddingModel); // AJOUTEZ CE FICHIER

        // === PHASE 2: RÉCUPÉRATION (avec Routage) ===

        // 1. Créer 2 ContentRetrievers
        ContentRetriever ragRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(ragStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .build();

        // On crée un retriever pour la finance
        ContentRetriever financeRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(financeStore) // Il pointe vers le bon magasin
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .build();

        // 2. Créer la Map de description pour le routage
        Map<ContentRetriever, String> retrieverMap = new HashMap<>();
        retrieverMap.put(ragRetriever, "Information sur le RAG (Retrieval-Augmented Generation) et l'intelligence artificielle");
        retrieverMap.put(financeRetriever, "Information sur la finance, l'économie, les banques et les investissements");

        // 3. Créer le QueryRouter
        // C'est ici que la magie opère : le LLM va lire les descriptions
        QueryRouter queryRouter = new LanguageModelQueryRouter(model, retrieverMap);

        // 4. Créer le RetrievalAugmentor (nouveau !)
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // 5. Créer l'assistant (en utilisant .retrievalAugmentor())
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor) // <-- NOUVELLE MÉTHODE
                .build();

        // Boucle de chat
        System.out.println("\nBonjour ! Posez vos questions sur le RAG ou les voitures.");
        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("\nVous : ");
            String question = scanner.nextLine();
            if (question.equalsIgnoreCase("stop")) {
                break;
            }
            String reponse = assistant.chat(question);
            System.out.println("Assistant : " + reponse);
        }
        scanner.close();
    }
}