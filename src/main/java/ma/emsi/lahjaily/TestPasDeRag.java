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
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.query.Query;
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
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class TestPasDeRag {

    // Helper pour récupérer le chemin du fichier de ressources
    private static Path getPath(String fileName) {
        try {
            URI fileUri = TestPasDeRag.class.getClassLoader().getResource(fileName).toURI();
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

    // Méthode helper pour l'ingestion
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

    /**
     * Classe interne pour notre routeur personnalisé, comme demandé dans le TP.
     * Elle implémente l'interface QueryRouter.
     * Cette version utilise le PromptTemplate (pour le bonus).
     */
    static class CustomQueryRouter implements QueryRouter {
        private final ChatModel chatModel;
        private final ContentRetriever ragRetriever;

        // Le template de prompt demandé, avec la variable {{query}}
        private final PromptTemplate promptTemplate = PromptTemplate.from(
                "Est-ce que la requête '{{query}}' porte sur l'IA (Intelligence Artificielle) ou le 'RAG' (Retrieval Augmented Generation) ? "
                        + "Réponds seulement par 'oui' ou 'non'."
        );

        // Constructeur pour injecter les dépendances (le modèle et le retriever)
        public CustomQueryRouter(ChatModel chatModel, ContentRetriever ragRetriever) {
            this.chatModel = chatModel;
            this.ragRetriever = ragRetriever;
        }

        @Override
        public Collection<ContentRetriever> route(Query query) {
            // 1. Créer le prompt en appliquant la question de l'utilisateur au template
            String prompt = promptTemplate.apply(Map.of("query", query.text())).text();

            // 2. Demander au LLM de décider (en utilisant .chat() au lieu de .generate())
            String decision = chatModel.chat(prompt);

            // 3. Analyser la décision
            if (decision.toLowerCase().contains("oui")) {
                System.out.println("LOG: Routage [RAG] activé.");
                return Collections.singletonList(ragRetriever);
            } else {
                System.out.println("LOG: Routage [Pas de RAG] activé. (Réponse LLM: " + decision + ")");
                return Collections.emptyList();
            }
        }
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

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // === PHASE 1: INGESTION (1 document) ===
        // On n'utilise que le rag.pdf pour ce test
        EmbeddingStore<TextSegment> ragStore = ingestDocument("rag.pdf", embeddingModel);

        // === PHASE 2: RÉCUPÉRATION (Routage personnalisé) ===

        // 1. Créer le ContentRetriever (un seul)
        ContentRetriever ragRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(ragStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .build();

        // 2. Créer une instance de notre QueryRouter personnalisé
        QueryRouter customRouter = new CustomQueryRouter(model, ragRetriever);

        // 3. Créer le RetrievalAugmentor
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(customRouter) // On utilise notre routeur
                .build();

        // 4. Créer l'assistant
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        // Boucle de chat
        System.out.println("\nBonjour ! Je réponds aux questions sur le RAG (et ignore le reste).");
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