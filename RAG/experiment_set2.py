EXPERIMENT_SET =[
        {
            "id": "Q_NIDD_use_06",
            "category": "usefulness",
            "sample_id": "null",
                    "xai_file": "none",
            "query": "Compare the performance of Random Forest, Multi-Layer Perceptron (MLP), and Naive Bayes on multiclass classification in the 5G-NIDD paper.",
            "ground_truth": "Random Forest achieved 99.48% accuracy (11.70 s training time) and MLP achieved the highest overall accuracy at 99.50% (316.88 s training time). Naive Bayes performed poorly across all evaluation metrics with an accuracy of only 40.99% (1.29 s training time)."
        },
        {
            "id": "Q_NIDD_use_08",
            "category": "usefulness",
            "sample_id": "null",
                    "xai_file": "none",
            "query": "What distinguishes the benign traffic generation methodology in 5G-NIDD from legacy datasets like CICIDS2017 or Bot-IoT?",
            "ground_truth": "5G-NIDD generated benign traffic by capturing live, real-time activity (streaming, browsing, SSH, SFTP) directly from real physical mobile devices attached to a functional 5G testbed. Legacy datasets predominantly relied on pre-collected simulated traffic, virtual instances, or post-processed artificial traffic injections."
        },
        {
            "id": "Q_NIDD_ret_09",
            "category": "retrieval",
            "sample_id": "null",
            "xai_file": "none",
            "query": "How do Slowloris and Torshammer attacks differ in execution at the application layer within the 5G-NIDD dataset?",
            "ground_truth": "Slowloris continuously transmits HTTP partial headers to keep connections open until server resources exhaust. Torshammer sends slow HTTP POST requests where the header specifies a large packet size, but the actual payload is transmitted extremely slowly, forcing the server to wait indefinitely for completion."
        },
        {
            "id": "Q_5GAD_ret_10",
            "category": "retrieval",
            "sample_id": "null",
            "xai_file": "none",
            "query": "What payload and HTTP methods are used in the FakeAMFInsert attack to register a rogue AMF instance with the NRF?",
            "ground_truth": "It uses an HTTP PUT request sending a JSON payload containing details like nfInstanceId (b01dface-bead-cafe-bade-cabledfabled), nfType (AMF), plmnList, sNssais, and amfInfo to the NRF management endpoint http://127.0.0.10:8000/nnrf-nfm/v1/nf-instances/<instance-id>."
        },
        {
            "id": "Q_5GAD_ret_11",
            "category": "retrieval",
            "sample_id": "null",
            "xai_file": "none",
            "query": "How does the randomDataDump attack exploit input validation vulnerabilities in the free5GC NRF service?",
            "ground_truth": "It sets the requester-nf-type query parameter to a random, unvalidated string during an nf-instances request. Due to a lack of input sanitization, the NRF still returns information for all registered network functions."
        },
        {
            "id": "Q_NIDD_use_12",
            "category": "usefulness",
            "sample_id": "null",
            "xai_file": "none",
            "query": "What network flow features rank highest according to ANOVA F-scores for binary classification versus multiclass classification in 5G-NIDD?",
            "ground_truth": "For binary classification, the top feature is Seq (Score: 329,589.08), followed by Offset and sTtl. For multiclass classification, the top feature is tcp (Score: 1,460,107.76), followed by AckDat (283,300.90) and sHops (159,802.03)"
        },
        {
            "id": "Q_5GAD_ret_13",
            "category": "retrieval",
            "sample_id": "null",
            "xai_file": "none",
            "query": "What are the memory and hardware limitations specified for preprocessing raw PCAP files in the 5GAD dataset scripts?",
            "ground_truth": "Unmodified processing with Data_prep.ipynb or Data_prep.py requires at least 96 GB of RAM and several hours of execution time, primarily because the sniff(...) function stores sequential packets in memory."
        }
]