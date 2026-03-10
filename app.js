(function () {
	// Phase sections
	const overviewSection = document.getElementById('overviewSection');
	const phase1Section = document.getElementById('phase1Section');
	const phase2Section = document.getElementById('phase2Section');
	const phase3Section = document.getElementById('phase3Section');
	const phase4Section = document.getElementById('phase4Section');
	
	// Navigation
	const phaseTabs = Array.prototype.slice.call(document.querySelectorAll('.phase-tab'));
	
	// Dataset control
	const datasetSelect = document.getElementById('datasetSelect');
	
	// Modal elements
	const modal = document.getElementById('imageModal');
	const modalImg = document.getElementById('modalImg');
	const closeBtn = document.querySelector('.close');
	
	// Phase 1 elements
	const phase1ImportanceImg = document.getElementById('phase1ImportanceImg');
	const phase1ImportanceMsg = document.getElementById('phase1ImportanceMsg');
	const phase1BeeswarmImg = document.getElementById('phase1BeeswarmImg');
	const phase1BeeswarmMsg = document.getElementById('phase1BeeswarmMsg');
	
	// Phase 2 elements
	const phase2ModelSelect = document.getElementById('phase2ModelSelect');
	const phase2ConfusionImg = document.getElementById('phase2ConfusionImg');
	const phase2ConfusionMsg = document.getElementById('phase2ConfusionMsg');
	const phase2ShapSummaryImg = document.getElementById('phase2ShapSummaryImg');
	const phase2ShapSummaryMsg = document.getElementById('phase2ShapSummaryMsg');
	const phase2IndexSelect = document.getElementById('phase2IndexSelect');
	const phase2ForcePlotImg = document.getElementById('phase2ForcePlotImg');
	const phase2ForcePlotMsg = document.getElementById('phase2ForcePlotMsg');
	const phase2LimeImg = document.getElementById('phase2LimeImg');
	const phase2LimeMsg = document.getElementById('phase2LimeMsg');
	
	// Phase 3 elements
	const phase3ClassSelect = document.getElementById('phase3ClassSelect');
	const phase3ConfusionImg = document.getElementById('phase3ConfusionImg');
	const phase3ConfusionMsg = document.getElementById('phase3ConfusionMsg');
	const phase3ImportanceImg = document.getElementById('phase3ImportanceImg');
	const phase3ImportanceMsg = document.getElementById('phase3ImportanceMsg');
	const phase3ClassBeeswarmImg = document.getElementById('phase3ClassBeeswarmImg');
	const phase3ClassBeeswarmMsg = document.getElementById('phase3ClassBeeswarmMsg');
	
	// Phase 4 elements
	const phase4ModelSelect = document.getElementById('phase4ModelSelect');
	const phase4ClassSelect = document.getElementById('phase4ClassSelect');
	const phase4ConfusionImg = document.getElementById('phase4ConfusionImg');
	const phase4ConfusionMsg = document.getElementById('phase4ConfusionMsg');
	const phase4ImportanceImg = document.getElementById('phase4ImportanceImg');
	const phase4ImportanceMsg = document.getElementById('phase4ImportanceMsg');
	const phase4ClassBeeswarmImg = document.getElementById('phase4ClassBeeswarmImg');
	const phase4ClassBeeswarmMsg = document.getElementById('phase4ClassBeeswarmMsg');
	const phase4IndexSelect = document.getElementById('phase4IndexSelect');
	const phase4LimeImg = document.getElementById('phase4LimeImg');
	const phase4LimeMsg = document.getElementById('phase4LimeMsg');

	// Configuration
	const datasets = ['5G-NIDD', '5GAD', '5GC_PFCP'];
	const models = ['DecisionTree', 'DNN', 'MLP', 'RandomForest', 'XGBoost'];
	
	// Attack classes per dataset
	const attackClassesByDataset = {
		'5G-NIDD': ['HTTPFlood', 'ICMPFlood', 'SlowrateDoS', 'SYNFlood', 'SYNScan', 'TCPConnectScan', 'UDPFlood', 'UDPScan'],
		'5GAD': ['AMFLookingForUDM', 'CrashNRF', 'FakeAMFDelete', 'FakeAMFInsert', 'GetAllNFs', 'GetUserData', 'automatedDropWithTimer', 'automatedRedirectWithTimer', 'randomAMFInsert', 'randomDataDump'],
		'5GC_PFCP': ['Mal_Del', 'Mal_Estab', 'Mal_Mod', 'Mal_Mod2']
	};
	
	// Attack class display names (for better UI)
	const attackClassDisplayNames = {
		// 5G-NIDD
		'HTTPFlood': 'HTTP Flood',
		'ICMPFlood': 'ICMP Flood',
		'SlowrateDoS': 'Slowrate DoS',
		'SYNFlood': 'SYN Flood',
		'SYNScan': 'SYN Scan',
		'TCPConnectScan': 'TCP Connect Scan',
		'UDPFlood': 'UDP Flood',
		'UDPScan': 'UDP Scan',
		// 5GAD
		'AMFLookingForUDM': 'AMF Looking For UDM',
		'CrashNRF': 'Crash NRF',
		'FakeAMFDelete': 'Fake AMF Delete',
		'FakeAMFInsert': 'Fake AMF Insert',
		'GetAllNFs': 'Get All NFs',
		'GetUserData': 'Get User Data',
		'automatedDropWithTimer': 'Automated Drop With Timer',
		'automatedRedirectWithTimer': 'Automated Redirect With Timer',
		'randomAMFInsert': 'Random AMF Insert',
		'randomDataDump': 'Random Data Dump',
		// 5GC_PFCP
		'Mal_Del': 'Malicious Delete',
		'Mal_Estab': 'Malicious Establish',
		'Mal_Mod': 'Malicious Modify',
		'Mal_Mod2': 'Malicious Modify 2'
	};
	
	// Binary classification sample indices per dataset and model
	const binaryIndicesByDataset = {
		'5G-NIDD': {
			'DecisionTree': ['6822', '38037', '45204', '198673', '228693', '231936'],
			'DNN': ['7233', '63665', '79720', '106760', '133084', '166288'],
			'MLP': ['7233', '66672', '78504', '118835', '123511', '217502'],
			'RandomForest': ['7233', '36067', '38037', '82174', '118800', '147275'],
			'XGBoost': ['7233', '36067', '63663', '79718', '106759', '235852']
		},
		'5GAD': {
			'DecisionTree': ['733', '3987', '5419', '5742', '6721', '8080'],
			'DNN': ['733', '3808', '4684', '5612', '5742', '6721'],
			'MLP': ['733', '3808', '5420', '5712', '5870', '8081'],
			'RandomForest': ['2999', '4016', '5371', '7159', '7193', '7915'],
			'XGBoost': ['733', '3808', '5420', '5712', '5870', '8081']
		},
		'5GC_PFCP': {
			'DecisionTree': ['8', '1916', '2644', '3409', '3872', '4844'],
			'DNN': ['94', '1115', '1826', '3305', '3729', '4760'],
			'MLP': ['2323', '4045', '4107', '4207', '4760', '5115'],
			'RandomForest': ['1384', '1411', '2147', '2399', '4886', '5344'],
			'XGBoost': ['8', '668', '3029', '3334', '3405', '5314']
		}
	};
	
	// Multi-class sample indices per dataset and model
	const multiclassIndicesByDataset = {
		'5G-NIDD': {
			'DecisionTree': ['17958', '34284', '49029', '62902', '72269', '99932'],
			'DNN': ['51053', '76430', '80091', '95609', '134168', '134532'],
			'MLP': ['43659', '115556', '121708', '134307', '139723', '143734'],
			'RandomForest': ['13634', '20419', '85706', '90126', '116224', '127844'],
			'XGBoost': ['6335', '55461', '77417', '82590', '89106', '101027']
		},
		'5GAD': {
			'DecisionTree': ['746', '1550', '4530'],
			'DNN': ['200', '801', '1660', '1875', '3010', '4404'],
			'MLP': ['34', '560', '1567', '2306', '4385', '4749'],
			'RandomForest': ['746', '1550', '4530'],
			'XGBoost': ['746', '1550', '4530']
		},
		'5GC_PFCP': {
			'DecisionTree': ['2720', '2754', '4044', '4086', '4280', '4284'],
			'DNN': ['1001', '1367', '1596', '2394', '3041', '3119'],
			'MLP': ['220', '2400', '3269', '3383', '3667', '4139'],
			'RandomForest': ['903', '1770', '2351', '2492', '3009', '3441'],
			'XGBoost': ['628', '786', '997', '2119', '2747', '4255']
		}
	};

	function populateSelect(select, options, displayNames) {
		select.innerHTML = '';
		options.forEach(function (opt) {
			const option = document.createElement('option');
			option.value = opt;
			option.textContent = displayNames && displayNames[opt] ? displayNames[opt] : opt;
			select.appendChild(option);
		});
	}
	
	function getAttackClasses() {
		const dataset = getCurrentDataset();
		return attackClassesByDataset[dataset] || attackClassesByDataset['5G-NIDD'];
	}

	function getCurrentDataset() {
		return datasetSelect.value;
	}

	function getDatasetBasePath() {
		const dataset = getCurrentDataset();
		return 'results/' + dataset;
	}

	function getBinaryIndices(model) {
		const dataset = getCurrentDataset();
		if (!model) model = 'DecisionTree'; // Default
		return binaryIndicesByDataset[dataset][model] || binaryIndicesByDataset['5G-NIDD'][model];
	}

	function getMulticlassIndices(model) {
		const dataset = getCurrentDataset();
		return multiclassIndicesByDataset[dataset][model] || multiclassIndicesByDataset['5G-NIDD'][model];
	}

	function normalizeBinaryModelForFiles(model) {
		if (model === 'LightGBM') return 'lightgbm';
		return model;
	}

	function showImgOrMsg(imgEl, msgEl, src) {
		if (!src) {
			imgEl.style.display = 'none';
			msgEl.style.display = 'block';
			return;
		}
		imgEl.onload = function () {
			imgEl.style.display = 'block';
			msgEl.style.display = 'none';
		};
		imgEl.onerror = function () {
			imgEl.style.display = 'none';
			msgEl.style.display = 'block';
		};
		imgEl.src = src;
	}

	// Function to load and render markdown
	function loadMarkdown(elementId, mdPath) {
		const container = document.getElementById(elementId);
		if (!container) return;
		
		fetch(mdPath)
			.then(response => {
				if (!response.ok) throw new Error('Markdown file not found');
				return response.text();
			})
			.then(mdContent => {
				const rawHtml = marked.parse(mdContent);
				const cleanHtml = DOMPurify.sanitize(rawHtml);
				container.innerHTML = cleanHtml;
			})
			.catch(err => {
				console.error('[Markdown] failed:', err);
				container.innerHTML = '<p style="color:#999;">Content not available.</p>';
			});
	}

	// Function to load and display overall score from JSON
	function loadOverallScore(elementId, jsonPath, sampleIndex) {
		const container = document.getElementById(elementId);
		if (!container) return;
		
		fetch(jsonPath)
			.then(response => {
				if (!response.ok) throw new Error('JSON file not found');
				return response.json();
			})
			.then(data => {
				// Find the sample with matching index
				const sample = data.find(function(item) {
					return String(item.sample_index) === String(sampleIndex);
				});
				
				if (sample && sample.overall_score !== undefined) {
					const score = (sample.overall_score * 100).toFixed(2);
					const scoreClass = sample.overall_score >= 0.7 ? 'score-high' : 
					                   sample.overall_score >= 0.4 ? 'score-medium' : 'score-low';
					container.innerHTML = '<strong>XAI Quality Score:</strong> <span class="' + scoreClass + '">' + score + '%</span>';
					container.style.display = 'block';
				} else {
					container.style.display = 'none';
				}
			})
			.catch(err => {
				console.error('[Score] failed:', err);
				container.style.display = 'none';
			});
	}

	// Function to load overall score from Markdown report file
	function loadGlobalScore(elementId, mdPath) {
		const container = document.getElementById(elementId);
		if (!container) return;
		
		fetch(mdPath)
			.then(response => {
				if (!response.ok) throw new Error('Markdown file not found');
				return response.text();
			})
			.then(mdContent => {
				// Extract overall score from markdown
				// Format: "## Overall Score: 0.732"
				const scoreMatch = mdContent.match(/##\s*Overall Score:\s*([\d.]+)/i);
				if (scoreMatch && scoreMatch[1]) {
					const scoreValue = parseFloat(scoreMatch[1]);
					const score = (scoreValue * 100).toFixed(2);
					const scoreClass = scoreValue >= 0.7 ? 'score-high' : 
					                   scoreValue >= 0.4 ? 'score-medium' : 'score-low';
					container.innerHTML = '<strong>XAI Quality Score:</strong> <span class="' + scoreClass + '">' + score + '%</span>';
					container.style.display = 'block';
				} else {
					container.style.display = 'none';
				}
			})
			.catch(err => {
				console.error('[Global Score] failed:', err);
				container.style.display = 'none';
			});
	}

	// Function to load and display LIME HTML into a stable container
	function showLimeHtmlInContainer(containerEl, htmlPath) {
		if (!containerEl) {
			console.error('LIME container element not found');
			return;
		}
		console.log('[LIME] render into container:', containerEl.id, 'path:', htmlPath);
		containerEl.innerHTML = '';
		fetch(htmlPath)
			.then(response => {
				if (!response.ok) throw new Error('HTML file not found');
				return response.text();
			})
			.then(htmlContent => {
				const iframe = document.createElement('iframe');
				iframe.srcdoc = htmlContent;
				iframe.style.width = '100%';
				iframe.style.height = '500px';
				iframe.style.border = '1px solid #e5e7eb';
				iframe.style.borderRadius = '4px';
				iframe.style.background = '#fff';
				iframe.style.overflow = 'auto';
				containerEl.innerHTML = '';
				containerEl.appendChild(iframe);
			})
			.catch(err => {
				console.error('[LIME] failed:', err);
				containerEl.innerHTML = '<p class="media-msg" style="display:block;">LIME explanation not found.</p>';
			});
	}

	// Modal functions
	function openModal(imgSrc) {
		modal.style.display = 'block';
		modalImg.src = imgSrc;
	}

	function closeModal() {
		modal.style.display = 'none';
	}

	// Add click event to all zoomable images
	function addZoomEvents() {
		const zoomableImages = document.querySelectorAll('.zoomable');
		zoomableImages.forEach(function(img) {
			img.addEventListener('click', function() {
				if (this.src && this.style.display !== 'none') {
					openModal(this.src);
				}
			});
		});
	}

	// Phase 1: Initial Processing (LightGBM)
	function updatePhase1() {
		const basePath = getDatasetBasePath();
		
		// Model results
		const confusionPath = basePath + '/binary/LightGBM/confusion_matrix.png';
		const clsReportPath = basePath + '/binary/LightGBM/classification_report.png';
		const phase1ConfusionImg = document.getElementById('phase1ConfusionImg');
		const phase1ConfusionMsg = document.getElementById('phase1ConfusionMsg');
		const phase1ClsReportImg = document.getElementById('phase1ClsReportImg');
		const phase1ClsReportMsg = document.getElementById('phase1ClsReportMsg');
		if (phase1ConfusionImg) showImgOrMsg(phase1ConfusionImg, phase1ConfusionMsg, confusionPath);
		if (phase1ClsReportImg) showImgOrMsg(phase1ClsReportImg, phase1ClsReportMsg, clsReportPath);
		
		// XAI visualizations
		const importanceBarPath = basePath + '/binary/LightGBM/shap_importance_bar.png';
		const beeswarmPath = basePath + '/binary/LightGBM/shap_beeswarm.png';
		showImgOrMsg(phase1ImportanceImg, phase1ImportanceMsg, importanceBarPath);
		showImgOrMsg(phase1BeeswarmImg, phase1BeeswarmMsg, beeswarmPath);
		
		// Radar chart
		const radarPath = basePath + '/binary/LightGBM/evaluate/shap_global/LightGBM_shap_global_evaluation_radar.png';
		const phase1ShapGlobalRadarImg = document.getElementById('phase1ShapGlobalRadarImg');
		const phase1ShapGlobalRadarMsg = document.getElementById('phase1ShapGlobalRadarMsg');
		if (phase1ShapGlobalRadarImg) showImgOrMsg(phase1ShapGlobalRadarImg, phase1ShapGlobalRadarMsg, radarPath);
		
		// Global Score
		const globalScorePath = basePath + '/binary/LightGBM/evaluate/shap_global/LightGBM_shap_global_evaluation_report.md';
		loadGlobalScore('phase1ShapGlobalScore', globalScorePath);
		
		// LLM explanation
		const llmPath = basePath + '/binary/LightGBM/LLM/global_explanation.md';
		loadMarkdown('phase1LLMGlobal', llmPath);
		
		setTimeout(addZoomEvents, 100);
	}

	// Phase 2: Binary Classification
	function updatePhase2() {
		const basePath = getDatasetBasePath();
		const model = phase2ModelSelect.value;
		const modelKey = normalizeBinaryModelForFiles(model);
		
		// Model results
		const confusionPath = basePath + '/binary/' + model + '/confusion_matrix.png';
		const clsReportPath = basePath + '/binary/' + model + '/classification_report.png';
		const phase2ClsReportImg = document.getElementById('phase2ClsReportImg');
		const phase2ClsReportMsg = document.getElementById('phase2ClsReportMsg');
		showImgOrMsg(phase2ConfusionImg, phase2ConfusionMsg, confusionPath);
		if (phase2ClsReportImg) showImgOrMsg(phase2ClsReportImg, phase2ClsReportMsg, clsReportPath);
		
		// SHAP Summary plot
		const shapSummaryPath = basePath + '/binary/' + model + '/shap_summary_plot.png';
		showImgOrMsg(phase2ShapSummaryImg, phase2ShapSummaryMsg, shapSummaryPath);
		
		// Global radar
		const radarPath = basePath + '/binary/' + model + '/evaluate/shap_global/' + model + '_shap_global_evaluation_radar.png';
		const phase2ShapGlobalRadarImg = document.getElementById('phase2ShapGlobalRadarImg');
		const phase2ShapGlobalRadarMsg = document.getElementById('phase2ShapGlobalRadarMsg');
		if (phase2ShapGlobalRadarImg) showImgOrMsg(phase2ShapGlobalRadarImg, phase2ShapGlobalRadarMsg, radarPath);
		
		// Global Score
		const globalScorePath = basePath + '/binary/' + model + '/evaluate/shap_global/' + model + '_shap_global_evaluation_report.md';
		loadGlobalScore('phase2ShapGlobalScore', globalScorePath);
		
		// LLM explanation
		const llmPath = basePath + '/binary/' + model + '/LLM/global_explanation.md';
		loadMarkdown('phase2LLMGlobal', llmPath);
	}

	function updatePhase2LocalXAI() {
		const basePath = getDatasetBasePath();
		const model = phase2ModelSelect.value;
		const index = phase2IndexSelect.value;
		console.log('=== updatePhase2LocalXAI ===', { model, index });
		
		// SHAP Force plot
		const forcePlotPath = basePath + '/binary/' + model + '/shap_force_plot_' + index + '.png';
		showImgOrMsg(phase2ForcePlotImg, phase2ForcePlotMsg, forcePlotPath);
		
		// SHAP Local Radar
		const shapRadarImg = document.getElementById('phase2ShapLocalRadarImg');
		const shapRadarMsg = document.getElementById('phase2ShapLocalRadarMsg');
		const binaryIndices = getBinaryIndices(model);
		const sampleNum = binaryIndices.indexOf(index) + 1;
		if (sampleNum > 0 && shapRadarImg) {
			const actualRadarPath = basePath + '/binary/' + model + '/evaluate/shap_local_individual/sample_' + sampleNum + '_idx_' + index + '_';
			// Load image with fallback handling - avoid infinite loop
			const img = new Image();
			img.onload = function() {
				shapRadarImg.src = img.src;
				shapRadarImg.style.display = 'block';
				if (shapRadarMsg) shapRadarMsg.style.display = 'none';
			};
			img.onerror = function() {
				// Try incorrect version once
				const incorrectPath = actualRadarPath + 'incorrect_radar.png';
				const img2 = new Image();
				img2.onload = function() {
					shapRadarImg.src = img2.src;
					shapRadarImg.style.display = 'block';
					if (shapRadarMsg) shapRadarMsg.style.display = 'none';
				};
				img2.onerror = function() {
					// Both failed, show message
					shapRadarImg.style.display = 'none';
					if (shapRadarMsg) shapRadarMsg.style.display = 'block';
				};
				img2.src = incorrectPath;
			};
			img.src = actualRadarPath + 'correct_radar.png';
		}
		
	// SHAP Overall Score
	const shapScorePath = basePath + '/binary/' + model + '/evaluate/shap_local_individual/' + model + '_shap_local_individual_results.json';
	loadOverallScore('phase2ShapLocalScore', shapScorePath, index);
	
	// LIME
	const limeHtmlPath = basePath + '/binary/' + model + '/lime_explanation_' + index + '.html';
	const limeContainer = document.getElementById('phase2LimeContainer');
	if (!limeContainer) { console.error('Phase 2 LIME container not found'); return; }
	showLimeHtmlInContainer(limeContainer, limeHtmlPath);
	
	// LIME Local Radar
	const limeRadarImg = document.getElementById('phase2LimeLocalRadarImg');
	const limeRadarMsg = document.getElementById('phase2LimeLocalRadarMsg');
	if (sampleNum > 0 && limeRadarImg) {
		const actualLimeRadarPath = basePath + '/binary/' + model + '/evaluate/lime_local_individual/sample_' + sampleNum + '_idx_' + index + '_';
		const limeImg = new Image();
		limeImg.onload = function() {
			limeRadarImg.src = limeImg.src;
			limeRadarImg.style.display = 'block';
			if (limeRadarMsg) limeRadarMsg.style.display = 'none';
		};
		limeImg.onerror = function() {
			const incorrectPath = actualLimeRadarPath + 'incorrect_radar.png';
			const limeImg2 = new Image();
			limeImg2.onload = function() {
				limeRadarImg.src = limeImg2.src;
				limeRadarImg.style.display = 'block';
				if (limeRadarMsg) limeRadarMsg.style.display = 'none';
			};
			limeImg2.onerror = function() {
				limeRadarImg.style.display = 'none';
				if (limeRadarMsg) limeRadarMsg.style.display = 'block';
			};
			limeImg2.src = incorrectPath;
		};
		limeImg.src = actualLimeRadarPath + 'correct_radar.png';
	}
	
	// LIME Overall Score
	const limeScorePath = basePath + '/binary/' + model + '/evaluate/lime_local_individual/' + model + '_lime_local_individual_results.json';
	loadOverallScore('phase2LimeLocalScore', limeScorePath, index);
	
	// Local LLM explanation (unified for both SHAP and LIME)
	const shapLlmPath = basePath + '/binary/' + model + '/LLM/individual_samples/sample_' + index + '.md';
	loadMarkdown('phase2LLMSampleLocal', shapLlmPath);
		
		setTimeout(addZoomEvents, 100);
	}

	// Phase 3: Multi-class Classification (LightGBM)
	function updatePhase3() {
		const basePath = getDatasetBasePath();
		const class_name = phase3ClassSelect.value;
		
		// Model results
		const confusionPath = basePath + '/multiclass/LightGBM_Multi/confusion_matrix.png';
		const clsReportPath = basePath + '/multiclass/LightGBM_Multi/classification_report.png';
		const phase3ClsReportImg = document.getElementById('phase3ClsReportImg');
		const phase3ClsReportMsg = document.getElementById('phase3ClsReportMsg');
		showImgOrMsg(phase3ConfusionImg, phase3ConfusionMsg, confusionPath);
		if (phase3ClsReportImg) showImgOrMsg(phase3ClsReportImg, phase3ClsReportMsg, clsReportPath);
		
		// Global importance
		const importancePath = basePath + '/multiclass/LightGBM_Multi/shap_importance_bar.png';
		showImgOrMsg(phase3ImportanceImg, phase3ImportanceMsg, importancePath);
		
		// Global radar
		const radarPath = basePath + '/multiclass/LightGBM_Multi/evaluate/shap_global/LightGBM_Multi_shap_global_evaluation_radar.png';
		const phase3ShapGlobalRadarImg = document.getElementById('phase3ShapGlobalRadarImg');
		const phase3ShapGlobalRadarMsg = document.getElementById('phase3ShapGlobalRadarMsg');
		if (phase3ShapGlobalRadarImg) showImgOrMsg(phase3ShapGlobalRadarImg, phase3ShapGlobalRadarMsg, radarPath);
		
		// Global Score
		const globalScorePath = basePath + '/multiclass/LightGBM_Multi/evaluate/shap_global/LightGBM_Multi_shap_global_evaluation_report.md';
		loadGlobalScore('phase3ShapGlobalScore', globalScorePath);
		
		// LLM explanation
		const llmPath = basePath + '/multiclass/LightGBM_Multi/LLM/global_explanation.md';
		loadMarkdown('phase3LLMGlobal', llmPath);
		
		// Class-specific beeswarm
		const beeswarmPath = basePath + '/multiclass/LightGBM_Multi/' + class_name + '_shap_beeswarm.png';
		showImgOrMsg(phase3ClassBeeswarmImg, phase3ClassBeeswarmMsg, beeswarmPath);
		
		setTimeout(addZoomEvents, 100);
	}

	// Phase 4: Multi-class Classification (Model Comparison)
	function updatePhase4() {
		const basePath = getDatasetBasePath();
		const model = phase4ModelSelect.value;
		
		// Model results
		const confusionPath = basePath + '/multiclass/' + model + '/confusion_matrix.png';
		const clsReportPath = basePath + '/multiclass/' + model + '/classification_report.png';
		const phase4ClsReportImg = document.getElementById('phase4ClsReportImg');
		const phase4ClsReportMsg = document.getElementById('phase4ClsReportMsg');
		showImgOrMsg(phase4ConfusionImg, phase4ConfusionMsg, confusionPath);
		if (phase4ClsReportImg) showImgOrMsg(phase4ClsReportImg, phase4ClsReportMsg, clsReportPath);
		
		// Global importance
		const importancePath = basePath + '/multiclass/' + model + '/shap_importance_bar.png';
		showImgOrMsg(phase4ImportanceImg, phase4ImportanceMsg, importancePath);
		
		// Global radar
		const radarPath = basePath + '/multiclass/' + model + '/evaluate/shap_global/' + model + '_shap_global_evaluation_radar.png';
		const phase4ShapGlobalRadarImg = document.getElementById('phase4ShapGlobalRadarImg');
		const phase4ShapGlobalRadarMsg = document.getElementById('phase4ShapGlobalRadarMsg');
		if (phase4ShapGlobalRadarImg) showImgOrMsg(phase4ShapGlobalRadarImg, phase4ShapGlobalRadarMsg, radarPath);
		
		// Global Score
		const globalScorePath = basePath + '/multiclass/' + model + '/evaluate/shap_global/' + model + '_shap_global_evaluation_report.md';
		loadGlobalScore('phase4ShapGlobalScore', globalScorePath);
		
		// LLM explanation
		const llmPath = basePath + '/multiclass/' + model + '/LLM/global_explanation.md';
		loadMarkdown('phase4LLMGlobal', llmPath);
	}

	function updatePhase4ClassSpecific() {
		const basePath = getDatasetBasePath();
		const model = phase4ModelSelect.value;
		const class_name = phase4ClassSelect.value;
		
		// Class-specific beeswarm
		const beeswarmPath = basePath + '/multiclass/' + model + '/' + class_name + '_shap_beeswarm.png';
		showImgOrMsg(phase4ClassBeeswarmImg, phase4ClassBeeswarmMsg, beeswarmPath);
	}

	function updatePhase4LocalXAI() {
		const basePath = getDatasetBasePath();
		const model = phase4ModelSelect.value;
		const index = phase4IndexSelect.value;
		console.log('=== updatePhase4LocalXAI ===', { model, index });
		
		// SHAP Waterfall plot (multiclass uses waterfall instead of force plot)
		const waterfallPath = basePath + '/multiclass/' + model + '/shap_waterfall_' + index + '.png';
		const phase4ForcePlotImg = document.getElementById('phase4ForcePlotImg');
		const phase4ForcePlotMsg = document.getElementById('phase4ForcePlotMsg');
		if (phase4ForcePlotImg) showImgOrMsg(phase4ForcePlotImg, phase4ForcePlotMsg, waterfallPath);
		
		// SHAP Local Radar
		const multiclassIndices = getMulticlassIndices(model);
		const sampleNum = multiclassIndices.indexOf(index) + 1;
		const shapRadarImg = document.getElementById('phase4ShapLocalRadarImg');
		const shapRadarMsg = document.getElementById('phase4ShapLocalRadarMsg');
		if (sampleNum > 0 && shapRadarImg) {
			const actualRadarPath = basePath + '/multiclass/' + model + '/evaluate/shap_local_individual/sample_' + sampleNum + '_idx_' + index + '_';
			const img = new Image();
			img.onload = function() {
				shapRadarImg.src = img.src;
				shapRadarImg.style.display = 'block';
				if (shapRadarMsg) shapRadarMsg.style.display = 'none';
			};
			img.onerror = function() {
				const incorrectPath = actualRadarPath + 'incorrect_radar.png';
				const img2 = new Image();
				img2.onload = function() {
					shapRadarImg.src = img2.src;
					shapRadarImg.style.display = 'block';
					if (shapRadarMsg) shapRadarMsg.style.display = 'none';
				};
				img2.onerror = function() {
					shapRadarImg.style.display = 'none';
					if (shapRadarMsg) shapRadarMsg.style.display = 'block';
				};
				img2.src = incorrectPath;
			};
			img.src = actualRadarPath + 'correct_radar.png';
		}
		
	// SHAP Overall Score
	const shapScorePath = basePath + '/multiclass/' + model + '/evaluate/shap_local_individual/' + model + '_shap_local_individual_results.json';
	loadOverallScore('phase4ShapLocalScore', shapScorePath, index);
	
	// LIME
	const limeHtmlPath = basePath + '/multiclass/' + model + '/lime_explanation_' + index + '.html';
	const limeContainer = document.getElementById('phase4LimeContainer');
	if (!limeContainer) { console.error('Phase 4 LIME container not found'); return; }
	showLimeHtmlInContainer(limeContainer, limeHtmlPath);
	
	// LIME Local Radar
	const limeRadarImg = document.getElementById('phase4LimeLocalRadarImg');
	const limeRadarMsg = document.getElementById('phase4LimeLocalRadarMsg');
	if (sampleNum > 0 && limeRadarImg) {
		const actualLimeRadarPath = basePath + '/multiclass/' + model + '/evaluate/lime_local_individual/sample_' + sampleNum + '_idx_' + index + '_';
		const limeImg = new Image();
		limeImg.onload = function() {
			limeRadarImg.src = limeImg.src;
			limeRadarImg.style.display = 'block';
			if (limeRadarMsg) limeRadarMsg.style.display = 'none';
		};
		limeImg.onerror = function() {
			const incorrectPath = actualLimeRadarPath + 'incorrect_radar.png';
			const limeImg2 = new Image();
			limeImg2.onload = function() {
				limeRadarImg.src = limeImg2.src;
				limeRadarImg.style.display = 'block';
				if (limeRadarMsg) limeRadarMsg.style.display = 'none';
			};
			limeImg2.onerror = function() {
				limeRadarImg.style.display = 'none';
				if (limeRadarMsg) limeRadarMsg.style.display = 'block';
			};
			limeImg2.src = incorrectPath;
		};
		limeImg.src = actualLimeRadarPath + 'correct_radar.png';
	}
	
	// LIME Overall Score
	const limeScorePath = basePath + '/multiclass/' + model + '/evaluate/lime_local_individual/' + model + '_lime_local_individual_results.json';
	loadOverallScore('phase4LimeLocalScore', limeScorePath, index);
	
	// Local LLM explanation (unified for both SHAP and LIME)
	const shapLlmPath = basePath + '/multiclass/' + model + '/LLM/individual_samples/sample_' + index + '.md';
	loadMarkdown('phase4LLMSampleLocal', shapLlmPath);
		
		setTimeout(addZoomEvents, 100);
	}

	function switchPhase(phaseKey) {
		// Update tab states
		phaseTabs.forEach(function (btn) { btn.classList.remove('active'); });
		const activeBtn = phaseTabs.find(function (b) { return b.getAttribute('data-phase') === phaseKey; });
		if (activeBtn) activeBtn.classList.add('active');

		// Hide all sections
		overviewSection.style.display = 'none';
		phase1Section.style.display = 'none';
		phase2Section.style.display = 'none';
		phase3Section.style.display = 'none';
		phase4Section.style.display = 'none';

		// Show active section and update content
		switch (phaseKey) {
			case 'overview':
				overviewSection.style.display = 'block';
				break;
			case 'phase1':
				phase1Section.style.display = 'block';
				updatePhase1();
				break;
			case 'phase2':
				phase2Section.style.display = 'block';
				updatePhase2();
				// Add delay to ensure elements are ready before updating local XAI
				setTimeout(updatePhase2LocalXAI, 100);
				break;
			case 'phase3':
				phase3Section.style.display = 'block';
				updatePhase3();
				break;
			case 'phase4':
				phase4Section.style.display = 'block';
				updatePhase4();
				updatePhase4ClassSpecific();
				// Add delay to ensure elements are ready
				setTimeout(updatePhase4LocalXAI, 100);
				break;
		}
	}

	function init() {
		// Initialize selects
		populateSelect(datasetSelect, datasets);
		populateSelect(phase2IndexSelect, getBinaryIndices('DecisionTree'));
		populateSelect(phase3ClassSelect, getAttackClasses(), attackClassDisplayNames);
		populateSelect(phase4ClassSelect, getAttackClasses(), attackClassDisplayNames);
		populateSelect(phase4IndexSelect, getMulticlassIndices('DecisionTree'));

		// Modal event listeners
		closeBtn.addEventListener('click', closeModal);
		modal.addEventListener('click', function(e) {
			if (e.target === modal) {
				closeModal();
			}
		});

		// Phase navigation
		phaseTabs.forEach(function (btn) {
			btn.addEventListener('click', function () {
				switchPhase(btn.getAttribute('data-phase'));
			});
		});

		// Phase 2 event listeners
		phase2ModelSelect.addEventListener('change', function() {
			console.log('Phase 2 model changed to:', this.value);
			// Update index options based on selected model and dataset
			populateSelect(phase2IndexSelect, getBinaryIndices(this.value));
			updatePhase2();
			// Add delay to ensure elements are ready
			setTimeout(updatePhase2LocalXAI, 100);
		});
		phase2IndexSelect.addEventListener('change', function() {
			console.log('Phase 2 index changed to:', this.value);
			// Add delay to ensure elements are ready
			setTimeout(updatePhase2LocalXAI, 50);
		});

		// Phase 3 event listeners
		phase3ClassSelect.addEventListener('change', updatePhase3);

		// Phase 4 event listeners
		phase4ModelSelect.addEventListener('change', function() {
			console.log('Phase 4 model changed to:', this.value);
			updatePhase4();
			updatePhase4ClassSpecific();
			// Update index options based on selected model and dataset
			populateSelect(phase4IndexSelect, getMulticlassIndices(phase4ModelSelect.value));
			// Add delay before updating local XAI to ensure elements are ready
			setTimeout(updatePhase4LocalXAI, 100);
		});
		phase4ClassSelect.addEventListener('change', updatePhase4ClassSpecific);
		phase4IndexSelect.addEventListener('change', function() {
			console.log('Phase 4 index changed to:', this.value);
			// Add delay to ensure elements are ready
			setTimeout(updatePhase4LocalXAI, 50);
		});

		// Dataset change - reload current phase and update indices
		datasetSelect.addEventListener('change', function() {
			console.log('Dataset changed to:', datasetSelect.value);
			// Update indices for the new dataset based on currently selected models
			populateSelect(phase2IndexSelect, getBinaryIndices(phase2ModelSelect.value));
			populateSelect(phase4IndexSelect, getMulticlassIndices(phase4ModelSelect.value));
			// Update attack classes for the new dataset
			populateSelect(phase3ClassSelect, getAttackClasses(), attackClassDisplayNames);
			populateSelect(phase4ClassSelect, getAttackClasses(), attackClassDisplayNames);
			// Find currently active phase and reload it
			const activeTab = phaseTabs.find(function (btn) { return btn.classList.contains('active'); });
			if (activeTab) {
				const currentPhase = activeTab.getAttribute('data-phase');
				switchPhase(currentPhase);
			}
		});

		// Start with overview
		switchPhase('overview');
	}

	if (document.readyState === 'loading') {
		document.addEventListener('DOMContentLoaded', init);
	} else {
		init();
	}
})();
