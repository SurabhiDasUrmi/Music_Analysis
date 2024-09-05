document.addEventListener("DOMContentLoaded", function() {
    // Fetch and display correlation matrix
    fetch('/api/correlation')
        .then(response => response.json())
        .then(data => {
            const correlationChart = document.getElementById('correlationChart');
            Plotly.newPlot(correlationChart, data, {
                responsive: true,
                scrollZoom: true,
                displayModeBar: true,
                displaylogo: false
            });
        });

    // Fetch and display positivity scores
    // fetch('/api/positivity')
    //     .then(response => response.json())
    //     .then(data => {
    //         Plotly.newPlot('positivity-plot', data);
    //     });

    // Fetch and display most popular genres
    // fetch('/api/genres')
    //     .then(response => response.json())
    //     .then(data => {
    //         Plotly.newPlot('genres-plot', JSON.parse(data));
    //     });

    // // Fetch and display clustering
    // // fetch('/api/clustering')
    // //     .then(response => response.json())
    // //     .then(data => {
    // //         Plotly.newPlot('clustering-plot', data);
    // //     });
    
    // fetch('/api/positivitychart')
    // .then(response => response.json())
    // .then(data => {
    //     const ctx = document.getElementById('positivityChart').getContext('2d');
    //     const chart = new Chart(ctx, {
    //         type: 'bar',
    //         data: {
    //             labels: data.countries,
    //             datasets: [{
    //                 label: 'Positivity Score',
    //                 data: data.scores,
    //                 backgroundColor: 'rgba(54, 162, 235, 0.2)',
    //                 borderColor: 'rgba(54, 162, 235, 1)',
    //                 borderWidth: 1
    //             }]
    //         },
    //         options: {
    //             responsive: true,
    //             scales: {
    //                 y: {
    //                     beginAtZero: true
    //                 }
    //             },
    //             animation: {
    //                 duration: 2000,
    //                 easing: 'easeInOutBounce'
    //             }
    //         }
    //     });
        
    // });


    fetch('/api/positivitychart')
    .then(response => response.json())
    .then(data => {
        const ctx = document.getElementById('positivityChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.countries,
                datasets: [{
                    label: 'Positivity Score',
                    data: data.scores,
                    backgroundColor: 'rgba(54, 162, 235, 0.7)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            display: false // Hide the grid lines
                        }
                    },
                    x: {
                        grid: {
                            display: false // Hide the grid lines
                        }
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeInOutBounce'
                }
            }
        });

        // Intersection Observer to trigger chart animation when it comes into view
        const chartObserver = new IntersectionObserver(entries => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    // Start the chart animation
                    chart.update(); // Ensure the chart updates to start animation
                    // Unobserve the target to avoid multiple animations
                    chartObserver.unobserve(entry.target);
                }
            });
        }, { threshold: 0.5 }); // Trigger when at least 50% of the chart is visible

        // Start observing the chart canvas
        chartObserver.observe(ctx.canvas);
    })
    .catch(error => {
        console.error('Error fetching data:', error);
    });



    fetch('/api/model-performance')
                .then(response => response.json())
                .then(data => {
                    const modelNames = data.map(d => d.model);
                    const accuracy = data.map(d => d.accuracy);
                    const precision = data.map(d => d.precision);
                    const recall = data.map(d => d.recall);
                    const f1_score = data.map(d => d.f1_score);

                    const ctx = document.getElementById('performanceChart').getContext('2d');
                    new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: modelNames,
                            datasets: [
                                {
                                    label: 'Accuracy',
                                    data: accuracy,
                                    backgroundColor: 'rgba(75, 192, 192, 0.6)',
                                    borderColor: 'rgba(75, 192, 192, 1)',
                                    borderWidth: 1
                                },
                                {
                                    label: 'Precision',
                                    data: precision,
                                    backgroundColor: 'rgba(153, 102, 255, 0.6)',
                                    borderColor: 'rgba(153, 102, 255, 1)',
                                    borderWidth: 1
                                },
                                {
                                    label: 'Recall',
                                    data: recall,
                                    backgroundColor: 'rgba(255, 159, 64, 0.6)',
                                    borderColor: 'rgba(255, 159, 64, 1)',
                                    borderWidth: 1
                                },
                                {
                                    label: 'F1 Score',
                                    data: f1_score,
                                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                                    borderColor: 'rgba(54, 162, 235, 1)',
                                    borderWidth: 1
                                }
                            ]
                        },
                        options: {
                            scales: {
                                y: {
                                    beginAtZero: true
                                }
                            },
                            animation: {
                                duration: 2000,
                                easing: 'easeInOutBounce'
                            }
                        }
                    });
                });
    

    fetch('/api/all-playlists')
    .then(response => response.json())
    .then(data => {
        const countrySelect = document.getElementById('countrySelect');
        Object.keys(data).forEach(country => {
            const option = document.createElement('option');
            option.value = country;
            option.textContent = country;
            countrySelect.appendChild(option);
        });
    });

    document.getElementById('playlistForm').addEventListener('submit', function(event) {
        event.preventDefault();
        const selectedCountry = document.getElementById('countrySelect').value;
        fetch(`/api/playlists?country=${selectedCountry}`)
            .then(response => response.json())
            .then(playlist => {
                const playlistContainer = document.getElementById('playlistContainer');
                playlistContainer.innerHTML = '';
                if (playlist.length > 0) {
                    playlist.forEach(song => {
                        const card = document.createElement('div');
                        card.className = 'col-md-4 mb-3';
                        card.innerHTML = `
                            <div class="card playlist-card">
                                <div class="card-body">
                                    <h5 class="card-title"><i class="fas fa-music"></i> ${song.name}</h5>
                                    <p class="card-text">${song.artists}</p>
                                    <p class="card-text"><small class="text-muted">Positivity Score: ${song.positivity_score.toFixed(2)}</small></p>
                                </div>
                            </div>
                        `;
                        playlistContainer.appendChild(card);
                    });

                    // const chartCanvas = document.createElement('canvas');
                    // chartCanvas.id = 'playlistChart';
                    // document.body.appendChild(chartCanvas);
                    document.getElementById('playlistChart').style.display = "block";
                    const ctx = document.getElementById('playlistChart').getContext('2d');
                    const scores = playlist.map(song => song.positivity_score);
                    const names = playlist.map(song => song.name);

                    new Chart(ctx, {
                        type: 'bar',
                        indexAxis: 'y',
                        data: {
                            labels: names,
                            datasets: [{
                                label: 'Positivity Score',
                                data: scores,
                                backgroundColor: 'rgba(122, 207, 246, 0.9)',
                                borderColor: 'rgba(75, 192, 192, 1)',
                                borderWidth: 1
                            }]
                        },
                        options: {
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    grid: {
                                        display: false // Hide the grid lines
                                    }
                                },
                                x: {
                                    grid: {
                                        display: false // Hide the grid lines
                                    }
                                }
                            },
                            animation: {
                                duration: 1000,
                                easing: 'easeInOutBounce'
                            }
                        }
                    });
                } else {
                    playlistContainer.innerHTML = '<div class="col-12 text-center text-white">No data available for the selected country.</div>';
                }
            });
    });


    fetch('/api/genres-map')
                .then(response => response.json())
                .then(mapData => {
                    Plotly.newPlot('mapContainer', mapData.data, mapData.layout);
                });



});