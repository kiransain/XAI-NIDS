EXPERIMENT_SET =[
        {
            "id": "Q_NIDD_control_new2",
            "category": "control",
            "sample_id": "null",
            "xai_file": "none",
            "query": "In a network intrusion dataset with roughly 1.2 million total flows, if about 477,000 are benign, what proportion of the traffic is attack traffic?",
            "ground_truth": "Roughly 61% is attack traffic — meaning attack traffic outnumbers benign traffic by a sizeable margin (~3:2). This matters for evaluation: a dataset this skewed toward attacks makes plain accuracy a weak metric, since a model predicting \"attack\" by default would already score well — precision/recall per class matter more here than in a balanced dataset."
        },
        {
            "id": "Q_NIDD_control_new3",
            "category": "control",
            "sample_id": "null",
            "xai_file": "none",
            "query": "If a feature like a sequence number ends up ranking as the most predictive feature in an intrusion detection model, what are two possible explanations — one benign, one concerning?",
            "ground_truth": "Benign explanation: if flows were collected in session order and attack sessions were run in separate contiguous blocks (a common data-collection pattern), sequence number can act as a rough time/session marker that correlates with attack periods for reasons unrelated to the traffic's actual content. Concerning explanation: this same mechanism is also the textbook definition of leakage — the model isn't learning attack behavior at all, it's learning \"which time period was this collected in,\" which would completely fail to generalize to a live deployment where sessions aren't neatly separated. The two explanations are actually the same underlying mechanism, just judged differently: whether it reflects the real world (deployment will always have this ordering) or an artifact of this specific data collection setup (deployment won't)."
        },
        {
            "id": "Q_NIDD_retadj_new",
            "category": "retrieval_adjacent",
            "sample_id": "null",
            "xai_file": "none",
            "query": "Between Decision Tree and Random Forest, which one tends to minimize false negatives and which tends to minimize false positives in typical intrusion detection benchmarks?",
            "ground_truth": "Random Forest averages predictions across many trees, which smooths out individual trees' overconfident misses — so it tends to catch more true attacks (fewer false negatives) at the cost of being slightly more trigger-happy (more false positives) than a single tree. A single Decision Tree, by contrast, makes one hard decision boundary per leaf — it can be more conservative about flagging attack (fewer false positives) but is more likely to miss attacks that fall just outside its learned rules (more false negatives). This is a general property of ensembling vs. single-tree models, not specific to one dataset."
        },
        {
            "id": "Q_NIDD_ret_01",
            "category": "retrieval",
            "sample_id": "null",
            "xai_file": "none",
            "query": "How were the target server and attacker nodes positioned relative to the Multi-access Edge Computing (MEC) environment and the base stations in the 5G-NIDD testbed?",
            "ground_truth": "The target server was an Ubuntu instance deployed inside the Multi-access Edge Computing (MEC) environment connected via the S1-U interface. The attacker nodes (Raspberry Pi 4 devices running Ubuntu) were connected via Wi-Fi to Huawei 5G modems, which attached over 5G to two Nokia Flexi Zone Indoor Pico Base Stations outside the MEC."
        },
        {
            "id": "Q_NIDD_ret_02",
            "category": "retrieval",
            "sample_id": "null",
            "xai_file": "none",
            "query": "Why was the GTP-U layer removed during the post-processing phase of the 5G-NIDD dataset, and which tool was used to accomplish this?",
            "ground_truth": "The GTP-U (GPRS Tunnelling Protocol User Plane) layer was removed because capturing packets directly from the radio access network (RAN) interface included tunneling headers used between the RAN and core network, which interferes with extracting standard flow features. The removal was performed using Tracewrangler."
        },
        {
            "id": "Q_5GAD_ret_03",
            "category": "retrieval",
            "sample_id": "null",
            "xai_file": "none",
            "query": "Explain how REST API vulnerabilities in free5GC network functions (such as NRF and UDM) were exploited to execute reconnaissance attacks in the 5GAD dataset.",
            "ground_truth": "Attacks like AMFLookingForUDM and GetAllNFs exploit the fact that the Network Repository Function (NRF) does not verify whether the requester is actually an AMF or authenticated entity. An attacker issues simple HTTP GET requests via curl to NRF discovery endpoints (e.g., http://127.0.0.10:8000/nnrf-disc/v1/nf-instances), which unconditionally return metadata on target network functions."
        },
        {
            "id": "Q_5GAD_ret_04",
            "category": "retrieval",
            "sample_id": "null",
            "xai_file": "none",
            "query": "How do PFCP session modification attacks (e.g., automatedRedirectWithTimer) manipulate user plane traffic in a 5G core?",
            "ground_truth": "The attack sniffs network traffic during UE connection for a Packet Forwarding Control Protocol (PFCP) session establishment request. Upon discovering a victim UE address, it sends malicious PFCP session modification requests to the User Plane Function (UPF) containing the victim's session ID and Forwarding Action Rule ID (FARID) to alter traffic routing or drop packets."
        },
        {
            "id": "Q_NIDD_use_05",
            "category": "usefulness",
            "sample_id": "null",
            "xai_file": "none",
            "query": "Which statistical methods were applied to select the top 10 features from the 112 extracted Argus flow features in 5G-NIDD, and what threshold was set for Pearson correlation?",
            "ground_truth": "Pairwise Pearson correlation was computed to eliminate redundant features using a correlation threshold of 0.90 (removing the feature with lower target correlation). The remaining features were then ranked using ANOVA F-scores to select the top 10."
        },
        {
            "id": "Q_NIDD_use_06",
            "category": "usefulness",
            "sample_id": "null",
                    "xai_file": "none",
            "query": "Compare the performance of Random Forest, Multi-Layer Perceptron (MLP), and Naive Bayes on multiclass classification in the 5G-NIDD paper.",
            "ground_truth": "Random Forest achieved 99.48% accuracy (11.70 s training time) and MLP achieved the highest overall accuracy at 99.50% (316.88 s training time). Naive Bayes performed poorly across all evaluation metrics with an accuracy of only 40.99% (1.29 s training time)."
        },
        {
            "id": "Q_NIDD_use_07",
            "category": "usefulness",
           "sample_id": "null",
                   "xai_file": "none",
            "query": "Which specific attack types were most frequently misclassified as benign traffic or misclassified amongst each other in the 5G-NIDD experiments?",
            "ground_truth": "A considerable portion of UDP flood and UDP scan flows were incorrectly classified as benign traffic. Additionally, misclassifications occurred between HTTP flood and Slowrate DoS attacks due to shared application-layer characteristics."
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