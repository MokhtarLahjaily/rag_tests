package ma.emsi.lahjaily;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import java.net.URI;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Test 5 : RAG avec récupération sur le Web (Tavily)
 * Cette classe reprend le RAG naïf et y ajoute un second ContentRetriever
 * qui recherche des informations sur le Web en utilisant Tavily.
 * Un DefaultQueryRouter est utilisé pour envoyer la requête aux deux retrievers.
 */
public class TestWebRag { // Renommage de la classe pour le Test 5

    // Helper pour charger les ressources (identique à RagNaif)
    private static Path getPath(String fileName) {
        try {
            URI fileUri = TestWebRag.class.getClassLoader().getResource(fileName).toURI();
            return Paths.get(fileUri);
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }

    // Méthode pour configurer le logging (identique à RagNaif)
    private static void configureLogger() {
        // Configure le logger sous-jacent (java.util.logging)
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE); // Ajuster niveau
        // Ajouter un handler pour la console pour faire afficher les logs
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    public static void main(String[] args) {
        // ==== 1. APPEL DU CONFIGURATEUR DE LOGGER ====
        configureLogger();

        // 0. Créer le ChatModel
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .modelName("gemini-2.5-flash")
                .logRequestsAndResponses(true)
                .temperature(0.3)
                .build();

        // === PHASE 1: ENREGISTREMENT (Ingestion) ===
        // (Identique à RagNaif)
        System.out.println("Phase 1 : Démarrage de l'ingestion...");
        Path documentPath = getPath("rag.pdf");
        DocumentParser parser = new ApacheTikaDocumentParser();
        Document document = FileSystemDocumentLoader.loadDocument(documentPath, parser);
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        Response<List<Embedding>> embeddingsResponse = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = embeddingsResponse.content();
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);
        System.out.println("Phase 1 : Ingestion terminée. " + segments.size() + " segments ajoutés.");

        // === PHASE 2: UTILISATION (Récupération PDF + Web) ===

        // 1. Création du ContentRetriever pour le PDF (l'existant)
        ContentRetriever ragRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2) // On garde 2 résultats du PDF
                .minScore(0.5)
                .build();

        // 2. Création du WebSearchEngine (Tavily) - NOUVEAU
        String tavilyKey = System.getenv("TAVILY_KEY");
        if (tavilyKey == null || tavilyKey.isEmpty()) {
            System.err.println("Erreur : La variable d'environnement TAVILY_KEY n'est pas définie.");
            System.err.println("Veuillez l'obtenir sur https://tavily.com/ et la configurer.");
            return;
        }
        WebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavilyKey)
                .build();

        // 3. Création du ContentRetriever pour le Web - NOUVEAU
        ContentRetriever webRetriever = WebSearchContentRetriever.builder()
                .webSearchEngine(webSearchEngine)
                // maxResults par défaut est 3, ce qui est correct.
                .build();

        // 4. Création du QueryRouter (DefaultQueryRouter) - NOUVEAU
        // On lui passe les 2 ContentRetrievers
        QueryRouter queryRouter = new DefaultQueryRouter(ragRetriever, webRetriever);

        // 5. Création du RetrievalAugmentor - NOUVEAU
        // Il utilisera le QueryRouter pour interroger les deux sources
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // 6. Ajout de la mémoire (inchangé)
        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        // 7. Création de l'assistant (modifié pour utiliser l'augmentor)
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .retrievalAugmentor(retrievalAugmentor) // <-- NOUVELLE MÉTHODE
                .build();

        // 8. Boucle de chat interactive (REMPLACEMENT DE LA QUESTION UNIQUE)
        System.out.println("\n==================================================");
        System.out.println("Bonjour ! Posez vos questions (infos du PDF RAG + Web).");
        System.out.println("Tapez 'stop' pour quitter.");

        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.print("\nVous : ");
                String question = scanner.nextLine();
                if (question.equalsIgnoreCase("stop")) {
                    break;
                }
                String reponse = assistant.chat(question);
                System.out.println("Assistant : " + reponse);
            }
        }

        System.out.println("\nProgramme terminé.");
    }
}