EXPERIMENT_SET =[
     {
            "id": "Q_5GAD_faith_95",
            "category": "faithfulness",
            "sample_id": "95",
            "xai_file": "evaluation_dataset/shap_individual_95.json",
            "query": "Sample 95, 5G-AD dataset. Model predicted Benign, but the true label is Attack. Feature attributions pushing toward Benign: payload_mean (importance -0.18), payload_std (-0.17), unique_bytes (-0.16). Pushing toward Attack: tcp_window (-0.12, i.e. tcp_window > -0.15 pushed benign; tcp_flags > 0.84 pushed +0.10 toward attack). Explain why the model got this wrong.",
            "ground_truth": "This is a false negative. Three payload-distribution features (mean, std, unique byte count) all pointed toward Benign and outweighed the weaker Attack-direction signals from tcp_flags/tcp_window. The model over-relied on payload statistics that looked \"normal\" and under-weighted the anomalous TCP-layer signal, missing the attack."
        },
        {
            "id": "Q_5GAD_use_95_actionable",
            "category": "usefulness",
            "sample_id": "95",
            "xai_file": "evaluation_dataset/shap_individual_95.json",
            "query": "Sample 95 was predicted Benign but was actually an Attack. The features that most drove the benign prediction were payload_mean, payload_std, and unique_bytes. What would you do differently to catch cases like this?",
            "ground_truth": "A reasonable answer should note that relying heavily on payload-statistics features is risky when they can look \"normal\" for disguised attacks, and suggest giving more weight to protocol-level features (e.g. tcp_flags, tcp_window) or using an ensemble/threshold review for borderline cases rather than a single feature set."
        },
        {
            "id": "Q_5GAD_use_7915",
            "category": "usefulness",
            "sample_id": "7915",
            "xai_file": "evaluation_dataset/shap_individual_7915.json",
            "query": "Sample 7915 shows a SYN flag (tcp_flags=2.0) on port 8000 alongside high payload variation. What does this combination mean for a network connection, and how should an analyst respond?",
            "ground_truth": "A SYN flag indicates an attempt to initiate a TCP connection. In combination with unusual payload variation on port 8000, it may indicate abnormal activity, but this evidence alone is not sufficient to conclude that the service is unauthorized. The analyst should compare the traffic with the normal behavior of the service on port 8000, inspect the source and destination, connection frequency, handshake completion, and related server logs. If the connection is unexpected or repeated anomalously, the flow should be investigated further and appropriate controls such as blocking or rate limiting can be considered."
        },
        {
            "id": "Q_5GAD_use_95_window",
            "category": "usefulness",
            "sample_id": "95",
            "xai_file": "evaluation_dataset/shap_individual_95.json",
            "query": "In Sample 95, a large TCP window size (tcp_window=5.27) pushes the model toward predicting normal traffic, even though the sample is actually an attack. Why does a large window size trick the model, and how should an analyst handle this?",
            "ground_truth": "A large TCP window is common in legitimate high-throughput connections and therefore may have become associated with benign traffic during training. In this case, however, that association contributes to a false-negative prediction, showing that TCP window size should not be treated as strong evidence of benign behavior by itself. The analyst should examine it together with TCP flags, payload statistics, traffic rate, ports, flow duration, and other attack indicators. Repeated cases of this type may also indicate that the model or decision threshold should be revised so that one apparently benign feature cannot outweigh several suspicious signals."
        },
        {
            "id": "Q_NIDD_faith_69166",
            "category": "faithfulness",
            "sample_id": "69166",
            "xai_file": "evaluation_dataset/lime_individual_69166.txt",
            "query": "Sample 69166, 5G-NIDD dataset. Model predicted Attack, and that's correct. The strongest feature is sTtl (value 63, importance +0.25) — much larger than the next feature, sMeanPktSz (importance -0.15). What does this tell you about the prediction?",
            "ground_truth": "The prediction is driven overwhelmingly by one feature: sTtl. A TTL of 63 (one below the common default of 64) can occur normally, so a prediction resting mostly on this single feature is less robust than one supported by multiple agreeing features, even though it happens to be correct here"
        },
        {
            "id": "Q_NIDD_use_69166",
            "category": "usefulness",
            "sample_id": "69166",
            "xai_file": "evaluation_dataset/lime_individual_69166.txt",
            "query": "Sample 69166 also shows smaller contributions from TcpRtt (importance +0.06) and SynAck (importance +0.05), both weaker than sTtl (+0.25). Are these enough on their own to trust the prediction?",
            "ground_truth": "No — TcpRtt and SynAck contribute far less than sTtl and shouldn't be treated as independent confirming evidence; the prediction is still essentially resting on one dominant feature (sTtl), so additional weak signals don't meaningfully strengthen confidence."
        },
        {
            "id": "Q_5GAD_faith_7915",
            "category": "faithfulness",
            "sample_id": "7915",
            "xai_file": "evaluation_dataset/shap_individual_7915.json",
            "query": "Sample 7915, 5G-AD dataset. Model predicted Attack, and that's correct. Top features: payload_std (value 33.4, importance +0.149), payload_mean (value 76.2, importance +0.077), payload_max (value 121.0, importance +0.075), all pushing toward Attack. What does this combination suggest?",
            "ground_truth": "Three related payload-size/variance features agree and jointly support the Attack prediction, rather than one feature dominating. This makes the prediction more robust than a single-feature-driven one, and suggests the traffic has unusually large and variable payloads consistent with abnormal behavior."
        },
        {
            "id": "Q_NIDD_faith_228693",
            "category": "faithfulness",
            "sample_id": "228693",
            "xai_file": "evaluation_dataset/shap_individual_228693.json",
            "query": "Sample 228693, 5G-NIDD dataset. Model predicted Attack, and that's correct. The strongest feature is Seq (value 32870, importance +0.25), ahead of sTtl (value 63, importance +0.19). Is a sequence number a feature you'd expect to matter for detecting an attack?",
            "ground_truth": "Not typically — a raw sequence number is usually just an identifier or counter and shouldn't have a meaningful causal relationship with whether traffic is malicious. A high importance on Seq is a signal worth being cautious about, since it may reflect an artifact of how the data was collected/ordered rather than genuine attack behavior."
        },
        {
            "id": "Q_NIDD_control_new1",
            "category": "control",
            "sample_id": "null",
            "xai_file": "none",
            "query": "What tool is commonly used to remove a GTP-U layer from captured 5G network packets, and why would this step be necessary before analysis?",
            "ground_truth": "GTP-U wraps user-plane packets for transport between the radio access network and core, adding a layer of headers not present in the original traffic. If left in place, feature extraction tools would compute packet sizes, protocols, and header fields on the wrapped packet rather than the actual traffic — skewing every downstream feature. Removing it (e.g. via Tracewrangler) restores the packets to their pre-tunneling form so features reflect real traffic behavior, not artifacts of the 5G transport mechanism."
        },
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