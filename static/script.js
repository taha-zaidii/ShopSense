// API base URL
const API_BASE = '';

// Initialize status check on page load
document.addEventListener('DOMContentLoaded', function () {
    checkStatus();
    setInterval(checkStatus, 5000); // Update status every 5 seconds
});

// Check system status
async function checkStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/status`);
        const data = await response.json();

        updateStatusBadge(data);
        updateStats(data);
        updateMetrics(data);

    } catch (error) {
        console.error('Error checking status:', error);
        updateStatusBadge({ model_loaded: false });
    }
}

// Update status badge
function updateStatusBadge(data) {
    const statusText = document.getElementById('statusText');
    const statusDot = document.querySelector('.status-dot');

    if (data.model_loaded) {
        statusText.textContent = 'Model Ready';
        statusDot.style.background = 'var(--success)';
    } else {
        statusText.textContent = 'No Model';
        statusDot.style.background = 'var(--warning)';
    }
}

// Update statistics
function updateStats(data) {
    document.getElementById('numUsers').textContent = data.num_users?.toLocaleString() || '-';
    document.getElementById('numProducts').textContent = data.num_products?.toLocaleString() || '-';
    document.getElementById('datasetSize').textContent = data.dataset_size?.toLocaleString() || '-';

    if (data.metrics && data.metrics.MAE) {
        document.getElementById('maeScore').textContent = data.metrics.MAE.toFixed(4);
    } else {
        document.getElementById('maeScore').textContent = '-';
    }
}

// Update metrics panel
function updateMetrics(data) {
    if (data.metrics && Object.keys(data.metrics).length > 0) {
        document.getElementById('metricsPanel').style.display = 'block';
        document.getElementById('metricMAE').textContent = data.metrics.MAE?.toFixed(4) || '-';
        document.getElementById('metricMSE').textContent = data.metrics.MSE?.toFixed(4) || '-';
        document.getElementById('metricRMSE').textContent = data.metrics.RMSE?.toFixed(4) || '-';
    }
}

// Train model
async function trainModel() {
    const trainBtn = document.getElementById('trainBtn');
    const sampleSize = parseInt(document.getElementById('sampleSize').value);
    const epochs = parseInt(document.getElementById('epochs').value);
    const progressDiv = document.getElementById('trainingProgress');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');

    // Validation
    if (sampleSize < 1000 || sampleSize > 50000) {
        alert('Sample size must be between 1,000 and 50,000');
        return;
    }

    if (epochs < 1 || epochs > 20) {
        alert('Epochs must be between 1 and 20');
        return;
    }

    // Disable button and show progress
    trainBtn.disabled = true;
    trainBtn.textContent = '⏳ Training...';
    progressDiv.style.display = 'block';
    progressFill.style.width = '0%';
    progressText.textContent = 'Starting training...';

    try {
        const response = await fetch(`${API_BASE}/api/train`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                sample_size: sampleSize,
                epochs: epochs
            })
        });

        const data = await response.json();

        if (data.success) {
            // Animate progress to 100%
            progressFill.style.width = '100%';
            progressText.textContent = '✓ Training completed successfully!';
            progressText.style.color = 'var(--success)';

            // Update metrics
            if (data.metrics) {
                document.getElementById('metricsPanel').style.display = 'block';
                document.getElementById('metricMAE').textContent = data.metrics.MAE.toFixed(4);
                document.getElementById('metricMSE').textContent = data.metrics.MSE.toFixed(4);
                document.getElementById('metricRMSE').textContent = data.metrics.RMSE.toFixed(4);
            }

            // Check status to update UI
            setTimeout(() => {
                checkStatus();
                trainBtn.disabled = false;
                trainBtn.innerHTML = '<span class="btn-icon">⚡</span> Start Training';

                // Hide progress after a delay
                setTimeout(() => {
                    progressDiv.style.display = 'none';
                }, 3000);
            }, 1000);

        } else {
            progressText.textContent = '✗ Training failed: ' + data.error;
            progressText.style.color = 'var(--danger)';
            trainBtn.disabled = false;
            trainBtn.innerHTML = '<span class="btn-icon">⚡</span> Start Training';
        }

    } catch (error) {
        console.error('Error training model:', error);
        progressText.textContent = '✗ Error: ' + error.message;
        progressText.style.color = 'var(--danger)';
        trainBtn.disabled = false;
        trainBtn.innerHTML = '<span class="btn-icon">⚡</span> Start Training';
    }
}

// Get random user
async function getRandomUser() {
    try {
        const response = await fetch(`${API_BASE}/api/random_user`);
        const data = await response.json();

        if (data.success) {
            document.getElementById('userId').value = data.user_id;
        }
    } catch (error) {
        console.error('Error getting random user:', error);
    }
}

// Get recommendations
async function getRecommendations() {
    const recommendBtn = document.getElementById('recommendBtn');
    const userId = document.getElementById('userId').value;
    const topN = parseInt(document.getElementById('topN').value);
    const resultsPanel = document.getElementById('resultsPanel');
    const recommendationsGrid = document.getElementById('recommendationsGrid');

    // Validation
    if (topN < 1 || topN > 20) {
        alert('Number of recommendations must be between 1 and 20');
        return;
    }

    // Disable button
    recommendBtn.disabled = true;
    recommendBtn.innerHTML = '<span class="btn-icon">⏳</span> Loading...';

    // Show loading state
    resultsPanel.style.display = 'block';
    recommendationsGrid.innerHTML = '<div class="loading" style="grid-column: 1/-1; height: 100px; border-radius: 1rem;"></div>';

    try {
        const response = await fetch(`${API_BASE}/api/recommend`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                user_id: userId || null,
                top_n: topN
            })
        });

        const data = await response.json();

        if (data.success) {
            // Update user ID display
            document.getElementById('resultUserId').textContent = data.user_id;
            document.getElementById('userId').value = data.user_id;

            // Display recommendations
            displayRecommendations(data.recommendations);

            // Scroll to results
            resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        } else {
            recommendationsGrid.innerHTML = `
                <div style="grid-column: 1/-1; text-align: center; padding: 2rem; color: var(--danger);">
                    <h3>❌ Error</h3>
                    <p>${data.error}</p>
                    <p style="margin-top: 1rem; color: var(--text-secondary);">Please train the model first.</p>
                </div>
            `;
        }

    } catch (error) {
        console.error('Error getting recommendations:', error);
        recommendationsGrid.innerHTML = `
            <div style="grid-column: 1/-1; text-align: center; padding: 2rem; color: var(--danger);">
                <h3>❌ Error</h3>
                <p>${error.message}</p>
            </div>
        `;
    } finally {
        recommendBtn.disabled = false;
        recommendBtn.innerHTML = '<span class="btn-icon">🎁</span> Get Recommendations';
    }
}

// Display recommendations
function displayRecommendations(recommendations) {
    const recommendationsGrid = document.getElementById('recommendationsGrid');

    if (!recommendations || recommendations.length === 0) {
        recommendationsGrid.innerHTML = `
            <div style="grid-column: 1/-1; text-align: center; padding: 2rem;">
                <h3>No recommendations found</h3>
            </div>
        `;
        return;
    }

    recommendationsGrid.innerHTML = recommendations.map((rec, index) => `
        <div class="recommendation-card" style="animation: slideIn 0.5s ease ${index * 0.05}s both;">
            <div class="recommendation-rank">#${index + 1}</div>
            <div class="recommendation-product">
                📦 Product ${rec.product_id}
            </div>
            <div class="recommendation-rating">
                ⭐ ${rec.predicted_rating.toFixed(2)}
            </div>
            <div class="recommendation-label">Predicted Rating</div>
        </div>
    `).join('');
}

// Add CSS animation for cards sliding in
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
`;
document.head.appendChild(style);

// Run verification
async function runVerification() {
    const verifyBtn = document.getElementById('verifyBtn');
    const sampleUsers = parseInt(document.getElementById('sampleUsersVerify').value);
    const verificationResults = document.getElementById('verificationResults');
    const rankingMetricsGrid = document.getElementById('rankingMetricsGrid');

    // Validation
    if (sampleUsers < 10 || sampleUsers > 100) {
        alert('Sample users must be between 10 and 100');
        return;
    }

    // Disable button
    verifyBtn.disabled = true;
    verifyBtn.innerHTML = '<span class="btn-icon">⏳</span> Verifying...';

    // Show loading
    verificationResults.style.display = 'block';
    rankingMetricsGrid.innerHTML = '<div class="loading" style="grid-column: 1/-1; height: 100px; border-radius: 1rem;"></div>';

    try {
        const response = await fetch(`${API_BASE}/api/verify`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                sample_users: sampleUsers,
                threshold: 4.0
            })
        });

        const data = await response.json();

        if (data.success) {
            displayVerificationMetrics(data.metrics);
        } else {
            rankingMetricsGrid.innerHTML = `
                <div style="grid-column: 1/-1; text-align: center; padding: 2rem; color: var(--danger);">
                    <h3>❌ Error</h3>
                    <p>${data.error}</p>
                    <p style="margin-top: 1rem; color: var(--text-secondary);">Please train the model first.</p>
                </div>
            `;
        }

    } catch (error) {
        console.error('Error running verification:', error);
        rankingMetricsGrid.innerHTML = `
            <div style="grid-column: 1/-1; text-align: center; padding: 2rem; color: var(--danger);">
                <h3>❌ Error</h3>
                <p>${error.message}</p>
            </div>
        `;
    } finally {
        verifyBtn.disabled = false;
        verifyBtn.innerHTML = '<span class="btn-icon">✓</span> Run Verification';
    }
}

// Display verification metrics
function displayVerificationMetrics(metrics) {
    const rankingMetricsGrid = document.getElementById('rankingMetricsGrid');

    const metricsToShow = [
        { key: 'Precision@10', label: 'Precision@10', desc: 'Relevant in Top 10' },
        { key: 'Recall@10', label: 'Recall@10', desc: 'Coverage of Relevant' },
        { key: 'Hit@10', label: 'Hit Rate@10', desc: 'Any Hit in Top 10' },
        { key: 'NDCG@10', label: 'NDCG@10', desc: 'Ranking Quality' },
        { key: 'catalog_coverage', label: 'Coverage', desc: 'Products Recommended' },
        { key: 'evaluated_users', label: 'Users Eval.', desc: 'Users Evaluated' }
    ];

    rankingMetricsGrid.innerHTML = metricsToShow.map(m => {
        let value = metrics[m.key];
        let displayValue = '-';

        if (value !== undefined && value !== null) {
            if (m.key === 'catalog_coverage') {
                displayValue = value.toFixed(1) + '%';
            } else if (m.key === 'evaluated_users') {
                displayValue = Math.round(value);
            } else {
                displayValue = (value * 100).toFixed(1) + '%';
            }
        }

        return `
            <div class="metric-card">
                <div class="metric-label">${m.label}</div>
                <div class="metric-value">${displayValue}</div>
                <div class="metric-desc">${m.desc}</div>
            </div>
        `;
    }).join('');
}

// Compare random user
async function compareRandomUser() {
    const userCompareResults = document.getElementById('userCompareResults');
    const userCompareContent = document.getElementById('userCompareContent');

    // Show loading
    userCompareResults.style.display = 'block';
    userCompareContent.innerHTML = '<div class="loading" style="height: 150px; border-radius: 1rem;"></div>';

    try {
        const response = await fetch(`${API_BASE}/api/compare_user`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({})
        });

        const data = await response.json();

        if (data.success) {
            displayUserComparison(data.comparison);
        } else {
            userCompareContent.innerHTML = `
                <div style="text-align: center; padding: 2rem; color: var(--danger);">
                    <h3>❌ Error</h3>
                    <p>${data.error}</p>
                </div>
            `;
        }

    } catch (error) {
        console.error('Error comparing user:', error);
        userCompareContent.innerHTML = `
            <div style="text-align: center; padding: 2rem; color: var(--danger);">
                <h3>❌ Error</h3>
                <p>${error.message}</p>
            </div>
        `;
    }
}

// Display user comparison
function displayUserComparison(comparison) {
    const userCompareContent = document.getElementById('userCompareContent');

    let actualItemsHtml = comparison.actual_top_rated.slice(0, 5).map((item, i) => `
        <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
            <span>📦 Product ${item.product_id}</span>
            <span style="color: var(--warning);">⭐ ${item.rating.toFixed(1)}</span>
        </div>
    `).join('');

    let recommendedItemsHtml = comparison.recommendations.slice(0, 5).map((item, i) => `
        <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
            <span>📦 Product ${item.product_id}</span>
            <span style="color: var(--primary);">⭐ ${item.predicted_rating.toFixed(2)}</span>
        </div>
    `).join('');

    userCompareContent.innerHTML = `
        <div style="background: rgba(255,255,255,0.05); border-radius: 0.75rem; padding: 1rem; margin-bottom: 1rem;">
            <h4 style="color: var(--primary); margin-bottom: 0.5rem;">User ${comparison.user_id} Summary</h4>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 0.5rem;">
                <div style="text-align: center;">
                    <div style="color: var(--text-secondary); font-size: 0.8rem;">Total Ratings</div>
                    <div style="font-size: 1.2rem; font-weight: 600;">${comparison.total_ratings}</div>
                </div>
                <div style="text-align: center;">
                    <div style="color: var(--text-secondary); font-size: 0.8rem;">Avg Rating</div>
                    <div style="font-size: 1.2rem; font-weight: 600;">${comparison.avg_rating.toFixed(2)}</div>
                </div>
                <div style="text-align: center;">
                    <div style="color: var(--text-secondary); font-size: 0.8rem;">High Rated (≥4)</div>
                    <div style="font-size: 1.2rem; font-weight: 600;">${comparison.high_rated_items}</div>
                </div>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
            <div style="background: rgba(255,193,7,0.1); border-radius: 0.75rem; padding: 1rem;">
                <h4 style="color: var(--warning); margin-bottom: 0.75rem;">🎯 Actual Top Rated</h4>
                ${actualItemsHtml || '<p style="color: var(--text-secondary);">No high-rated items</p>'}
            </div>
            <div style="background: rgba(139,92,246,0.1); border-radius: 0.75rem; padding: 1rem;">
                <h4 style="color: var(--primary); margin-bottom: 0.75rem;">✨ Model Recommendations</h4>
                ${recommendedItemsHtml}
            </div>
        </div>
        
        <div style="margin-top: 1rem; padding: 0.75rem; background: rgba(34,197,94,0.1); border-radius: 0.5rem; text-align: center;">
            <span style="color: var(--success);">Predicted Avg: ${comparison.predicted_avg_rating.toFixed(2)} ⭐</span>
        </div>
    `;
}
