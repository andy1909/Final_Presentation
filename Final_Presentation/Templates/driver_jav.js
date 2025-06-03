const API_BASE_URL = "https://7fb2-2402-9d80-30d-1b10-7df7-4553-2581-d009.ngrok-free.app";

// Initialize the map centered on Ho Chi Minh City
const map = L.map('map', {
    zoomControl: false
}).setView([10.7769, 106.7009], 11);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);

L.control.zoom({
    position: 'topright'
}).addTo(map);

let stores = [];
let markers = [];
let searchMarker = null;
let routeLayers = [];
let routes = {};

const redIcon = L.icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
    shadowSize: [41, 41]
});

// Function to clear all markers
function clearAllMarkers() {
    markers.forEach((marker) => map.removeLayer(marker));
    markers = [];
}

// Function to clear search marker
function clearSearchMarker() {
    if (searchMarker) {
        map.removeLayer(searchMarker);
        searchMarker = null;
    }
}

// Function to clear route layers
function clearRouteLayers() {
    routeLayers.forEach(layer => map.removeLayer(layer));
    routeLayers = [];
}

// Function to find midpoint of a route segment
function getMidpoint(coordinates) {
    const [start, end] = coordinates;
    return [
        (start[1] + end[1]) / 2, // lat
        (start[0] + end[0]) / 2  // lng
    ];
}

// Function to normalize Vietnamese text by removing diacritics
function removeVietnameseDiacritics(str) {
    if (typeof str !== "string") return "";
    const diacriticsMap = [
        { base: "A", chars: "ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ" },
        { base: "E", chars: "ÈÉẸẺẼÊỀẾỆỂỄ" },
        { base: "I", chars: "ÌÍỊỈĨ" },
        { base: "O", chars: "ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ" },
        { base: "U", chars: "ÙÚỤỦŨƯỪỨỰỬỮ" },
        { base: "Y", chars: "ỲÝỴỶỸ" },
        { base: "D", chars: "Đ" },
        { base: "a", chars: "àáạảãâầấậẩẫăằắặẳẵ" },
        { base: "e", chars: "èéẹẻẽêềếệểễ" },
        { base: "i", chars: "ìíịỉĩ" },
        { base: "o", chars: "òóọỏõôồốộổỗơờớợởỡ" },
        { base: "u", chars: "ùúụủũưừứựửữ" },
        { base: "y", chars: "ỳýỵỷỹ" },
        { base: "d", chars: "đ" },
    ];
    let normalized = str.toLowerCase();
    for (const map of diacriticsMap) {
        for (const char of map.chars) {
            normalized = normalized.replace(new RegExp(char, "g"), map.base.toLowerCase());
        }
    }
    return normalized;
}

function normalizeVietnameseText(text) {
    return removeVietnameseDiacritics(text);
}

// Update store search input
function updateStoreSearch() {
    const storeSearch = document.getElementById('storeSearch');
    const searchSuggestions = document.getElementById('searchSuggestions');
    const searchArrow = document.getElementById('searchArrow');

    function updateSuggestions(query = "") {
        searchSuggestions.innerHTML = "";
        if (stores.length === 0) {
            const noData = document.createElement("div");
            noData.className = "search-suggestion";
            noData.textContent = "Chưa có dữ liệu cửa hàng";
            noData.style.cursor = "default";
            searchSuggestions.appendChild(noData);
            searchSuggestions.classList.add("active");
            return;
        }

        const normalizedQuery = normalizeVietnameseText(query);
        const suggestions = query.length > 0
            ? stores.filter((store) => normalizeVietnameseText(store.name).includes(normalizedQuery))
            : stores;

        if (suggestions.length > 0) {
            suggestions.forEach((store) => {
                const div = document.createElement("div");
                div.className = "search-suggestion";
                div.textContent = store.name;
                div.addEventListener("mousedown", (e) => {
                    e.preventDefault();
                    storeSearch.value = store.name;
                    searchSuggestions.classList.remove("active");
                    searchArrow.classList.remove("fa-chevron-up");
                    searchArrow.classList.add("fa-chevron-down");

                    clearSearchMarker();
                    searchMarker = L.marker([store.lat, store.lng])
                        .addTo(map)
                        .bindPopup(`<b>${store.name}</b>`)
                        .openPopup();
                    map.setView([store.lat, store.lng], 15);
                });
                searchSuggestions.appendChild(div);
            });
        } else {
            const noData = document.createElement("div");
            noData.className = "search-suggestion";
            noData.textContent = "Không tìm thấy cửa hàng";
            noData.style.cursor = "default";
            searchSuggestions.appendChild(noData);
        }
        searchSuggestions.classList.add("active");
    }

    storeSearch.addEventListener("input", () => {
        updateSuggestions(storeSearch.value);
        clearSearchMarker();
    });

    storeSearch.addEventListener("focus", () => {
        updateSuggestions(storeSearch.value);
    });

    document.addEventListener("click", (e) => {
        if (!storeSearch.contains(e.target) && !searchSuggestions.contains(e.target) && !searchArrow.contains(e.target)) {
            searchSuggestions.classList.remove("active");
            searchArrow.classList.remove("fa-chevron-up");
            searchArrow.classList.add("fa-chevron-down");
        }
    });

    searchArrow.addEventListener("click", (e) => {
        e.stopPropagation();
        if (searchSuggestions.classList.contains("active")) {
            searchSuggestions.classList.remove("active");
            searchArrow.classList.remove("fa-chevron-up");
            searchArrow.classList.add("fa-chevron-down");
        } else {
            updateSuggestions(storeSearch.value);
            searchSuggestions.classList.add("active");
            searchArrow.classList.remove("fa-chevron-down");
            searchArrow.classList.add("fa-chevron-up");
            storeSearch.focus();
        }
    });
}

// Update vehicle options
function updateVehicleOptions() {
    const vehicleSelect = document.getElementById('vehicleCode');
    vehicleSelect.innerHTML = '<option value="">Chọn mã xe</option>';

    if (Object.keys(routes).length === 0) {
        const noDataOption = document.createElement('option');
        noDataOption.value = '';
        noDataOption.textContent = 'Không có dữ liệu tuyến đường';
        noDataOption.disabled = true;
        vehicleSelect.appendChild(noDataOption);
        return;
    }

    // Add vehicle type to dropdown options
    Object.keys(routes).forEach((code) => {
        const route = routes[code];
        const option = document.createElement('option');
        option.value = code;
        // Display vehicle code with type (e.g., Xe 1 (1 tấn) or Xe 2 (1,3 tấn))
        const vehicleType = route.type === 0 
            ? '1 tấn' 
            : route.type === 1 
            ? '1,3 tấn' 
            : 'Không xác định';
        option.textContent = `${code} (${vehicleType})`;
        vehicleSelect.appendChild(option);
    });
}

// Process routes data from server
function processRoutesDataFromServer(rawRoutesData) {
    console.log("Bắt đầu processRoutesDataFromServer với dữ liệu:", rawRoutesData);
    if (!stores || stores.length === 0) {
        console.error("Dữ liệu cửa hàng (stores) phải được tải trước khi xử lý tuyến đường từ server.");
        return false;
    }
    routes = {};
    if (!Array.isArray(rawRoutesData)) {
        console.error("Dữ liệu tuyến đường nhận từ server không hợp lệ (không phải mảng).", rawRoutesData);
        return false;
    }
    let allValid = true;
    rawRoutesData.forEach((routeData) => {
        const vehicleCode = `Xe ${routeData.id}`;
        if (!routeData || typeof routeData !== "object" || !routeData.id || !routeData.route || !Array.isArray(routeData.route)) {
            console.warn(`Tuyến đường cho ${vehicleCode} từ server không hợp lệ, bỏ qua.`);
            return;
        }
        const routeIndices = routeData.route;
        if (routeIndices.length < 2) {
            console.warn(`Dữ liệu tuyến đường cho ${vehicleCode} không hợp lệ (ít hơn 2 điểm dừng), bỏ qua.`);
            return;
        }
        const startIndex = routeIndices[0];
        if (typeof startIndex !== "number" || startIndex < 0 || startIndex >= stores.length) {
            console.error(`startIndex ${startIndex} từ server không hợp lệ cho ${vehicleCode}. Số lượng cửa hàng: ${stores.length}. Bỏ qua tuyến này.`);
            allValid = false;
            return;
        }
        const segments = [];
        for (let i = 0; i < routeIndices.length - 1; i++) {
            const fromIndex = routeIndices[i];
            const toIndex = routeIndices[i + 1];
            if (
                typeof fromIndex !== "number" || fromIndex < 0 || fromIndex >= stores.length ||
                typeof toIndex !== "number" || toIndex < 0 || toIndex >= stores.length
            ) {
                console.error(`Chỉ số cửa hàng từ server không hợp lệ trong tuyến của ${vehicleCode}. Từ: ${fromIndex}, Đến: ${toIndex}. Bỏ qua đoạn này.`);
                allValid = false;
                continue;
            }
            segments.push({ from: stores[fromIndex].name, to: stores[toIndex].name });
        }
        if (segments.length === 0 && routeIndices.length >= 2) {
            console.warn(`Không có đoạn đường hợp lệ nào được tạo cho ${vehicleCode} từ dữ liệu server.`);
        }
        if (segments.length > 0) {
            routes[vehicleCode] = {
                id: routeData.id, // Store the route ID from JSON
                startIndex: startIndex,
                segments: segments,
                totalCost: parseFloat(routeData.cost),
                totalTime: parseFloat(routeData.time),
                type: routeData.type // Store the vehicle type from JSON
            };
        }
    });
    if (Object.keys(routes).length > 0 || allValid) {
        localStorage.setItem("routes", JSON.stringify(routes));
        console.log("Các tuyến đường đã được xử lý và lưu vào localStorage:", routes);
        return true;
    } else {
        console.error("Không có tuyến đường hợp lệ nào được xử lý từ dữ liệu server.");
        return false;
    }
}

// Load data from server
document.getElementById('loadDataButton').addEventListener('click', async () => {
    const loadingMessage = document.getElementById('loadingMessage');
    const successMessage = document.getElementById('successMessage');
    const errorMessage = document.getElementById('errorMessage');

    function handleErrorUI(message, errorObj = null) {
        console.error(message, errorObj || '');
        errorMessage.textContent = message + (errorObj ? `: ${errorObj.message}` : '');
        errorMessage.style.backgroundColor = '#f8d7da';
        errorMessage.style.color = '#721c24';
        errorMessage.style.display = 'block';
        loadingMessage.style.display = 'none';
        successMessage.style.display = 'none';
    }

    loadingMessage.textContent = 'Đang chuẩn bị tải dữ liệu từ server...';
    loadingMessage.style.backgroundColor = '#fff3cd';
    loadingMessage.style.color = '#856404';
    loadingMessage.style.display = 'block';
    successMessage.style.display = 'none';
    errorMessage.style.display = 'none';

    // Step 1: Fetch Excel file
    let excelArrayBuffer;
    try {
        loadingMessage.textContent = 'Đang tải file Excel từ server...';
        const excelResponse = await fetch(`${API_BASE_URL}/Uploads/input.xlsx`, {
            cache: 'no-store',
            headers: { 'ngrok-skip-browser-warning': 'true' }
        });
        if (!excelResponse.ok) {
            const errorText = await excelResponse.text();
            throw new Error(`Tải file Excel thất bại: ${excelResponse.status} - ${errorText}`);
        }
        excelArrayBuffer = await excelResponse.arrayBuffer();
    } catch (error) {
        handleErrorUI('Lỗi khi tải file Excel:', error);
        return;
    }

    // Step 2: Process Excel data
    try {
        loadingMessage.textContent = 'Đang xử lý dữ liệu cửa hàng...';
        const workbook = XLSX.read(excelArrayBuffer, { type: 'array' });
        const sheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[sheetName];
        const csvData = XLSX.utils.sheet_to_csv(worksheet);

        const parsedExcelData = await new Promise((resolve, reject) => {
            Papa.parse(csvData, {
                header: true,
                skipEmptyLines: true,
                transformHeader: header => header.trim(),
                transform: value => String(value || '').trim(),
                complete: results => {
                    if (results.errors.length > 0) {
                        reject(new Error(`Lỗi parse CSV: ${results.errors[0].message}`));
                    } else {
                        resolve(results.data);
                    }
                },
                error: err => reject(new Error(`Lỗi parse: ${err.message}`))
            });
        });

        stores = parsedExcelData
            .map(row => ({
                name: row['Địa chỉ cửa hàng'],
                lat: parseFloat(row['Vĩ độ']),
                lng: parseFloat(row['Kinh độ'])
            }))
            .filter(store => store.name && !isNaN(store.lat) && !isNaN(store.lng));

        if (stores.length === 0) {
            handleErrorUI('Không tìm thấy dữ liệu cửa hàng hợp lệ.');
            return;
        }
    } catch (error) {
        handleErrorUI('Lỗi xử lý Excel:', error);
        return;
    }

    // Step 3: Fetch routes data
    try {
        loadingMessage.textContent = 'Đang tải dữ liệu tuyến đường...';
        const routesResponse = await fetch(`${API_BASE_URL}/api/get-routes-from-json/`, {
            cache: 'no-store',
            headers: { 'ngrok-skip-browser-warning': 'true' }
        });
        if (!routesResponse.ok) {
            const errorText = await routesResponse.text();
            throw new Error(`Tải routes thất bại: ${routesResponse.status} - ${errorText}`);
        }
        const rawRoutesData = await routesResponse.json();
        if (!processRoutesDataFromServer(rawRoutesData)) {
            handleErrorUI('Lỗi xử lý dữ liệu tuyến đường.');
            return;
        }

        loadingMessage.style.display = 'none';
        successMessage.textContent = 'Dữ liệu cửa hàng và tuyến đường đã được tải thành công!';
        successMessage.style.backgroundColor = '#d4edda';
        successMessage.style.color = '#155724';
        successMessage.style.display = 'block';
        updateStoreSearch();
        updateVehicleOptions();
    } catch (error) {
        handleErrorUI('Lỗi khi tải routes:', error);
    }
});

// Load route for vehicle
document.getElementById('loadRoute').addEventListener('click', async () => {
    const vehicleCode = document.getElementById('vehicleCode').value;
    const routeInfo = document.getElementById('routeInfo');
    const errorMessage = document.getElementById('errorMessage');

    errorMessage.style.display = 'none';
    routeInfo.style.display = 'none';
    routeInfo.innerHTML = '';
    clearRouteLayers();
    clearAllMarkers();

    if (!vehicleCode) {
        errorMessage.textContent = 'Vui lòng chọn mã số xe!';
        errorMessage.style.backgroundColor = '#f8d7da';
        errorMessage.style.color = '#721c24';
        errorMessage.style.display = 'block';
        return;
    }

    const route = routes[vehicleCode];
    if (!route || !route.segments) {
        errorMessage.textContent = `Không tìm thấy lộ trình cho mã xe ${vehicleCode}!`;
        errorMessage.style.backgroundColor = '#f8d7da';
        errorMessage.style.color = '#721c24';
        errorMessage.style.display = 'block';
        return;
    }

    routeInfo.style.display = 'block';

    // Add route ID, vehicle type, total cost, and total time at the top of routeInfo
    const routeSummary = document.createElement("div");
    // Determine vehicle type text based on route.type
    const vehicleTypeText = route.type === 0 
        ? '- Dùng xe loại: 1 tấn, Kích thước xe tải (dài x rộng x cao): 3,1 x 1,6 x 1,7 m'
        : route.type === 1 
        ? '- Dùng xe loại: 1,3 tấn, Kích thước xe tải (dài x rộng x cao): 3,3 x 1,7 x 1,7 m'
        : 'Loại xe không xác định';
    routeSummary.innerHTML = `
        <p><b>- Tuyến đường cho xe số ${route.id}</b></p>
        <p><b>${vehicleTypeText}</b></p>
        <p><b>- Tổng chi phí: ${route.totalCost.toFixed(3)} VND</b></p>
        <p><b>- Tổng thời gian: ${route.totalTime} phút</b></p>
        <hr>
    `;
    routeInfo.appendChild(routeSummary);

    try {
        const colors = [
            '#ff0000', '#00ff00', '#0000ff', '#ff00ff', '#00ffff', '#ffa500', '#800080',
            '#ff4500', '#32cd32', '#1e90ff', '#ff69b4', '#00ced1', '#daa520', '#9932cc',
            '#ff6347', '#3cb371', '#4682b4', '#ff1493', '#20b2aa', '#b8860b'
        ];
        let bounds = L.latLngBounds();
        const apiKey = '5b3ce3597851110001cf6248cdbe7e84db0c408b86674e9bda1fc09b';
        const url = 'https://api.openrouteservice.org/v2/directions/driving-car/geojson';

        if (stores.length === 0 || route.startIndex === undefined || route.startIndex < 0 || route.startIndex >= stores.length) {
            errorMessage.textContent = 'Dữ liệu cửa hàng không đầy đủ hoặc startIndex không hợp lệ.';
            errorMessage.style.backgroundColor = '#f8d7da';
            errorMessage.style.color = '#721c24';
            errorMessage.style.display = 'block';
            routeInfo.style.display = 'none';
            return;
        }
        let currentLocation = stores[route.startIndex];
        const startLocation = currentLocation;
        bounds.extend([currentLocation.lat, currentLocation.lng]);

        const startMarker = L.marker([currentLocation.lat, currentLocation.lng])
            .addTo(map)
            .bindPopup(`<b>Điểm bắt đầu: ${currentLocation.name}</b>`);
        markers.push(startMarker);

        if (route.segments.length === 0) {
            routeInfo.innerHTML = "<p>Tuyến đường này không có chặng nào.</p>";
            map.setView([currentLocation.lat, currentLocation.lng], 13);
            return;
        }

        for (let i = 0; i < route.segments.length; i++) {
            const segment = route.segments[i];
            const destination = stores.find((store) => store.name === segment.to);

            if (!destination) {
                console.warn(`Không tìm thấy cửa hàng đích: ${segment.to}`);
                const legInfo = document.createElement("div");
                legInfo.innerHTML = `<p><b>Lỗi: ${segment.from} → ${segment.to} (Không tìm thấy cửa hàng đích)</b></p><hr>`;
                routeInfo.appendChild(legInfo);
                continue;
            }

            const isReturnToStart = destination.name === startLocation.name &&
                                    destination.lat === startLocation.lat &&
                                    destination.lng === startLocation.lng;

            const body = {
                coordinates: [
                    [currentLocation.lng, currentLocation.lat],
                    [destination.lng, destination.lat],
                ],
            };

            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Authorization': apiKey,
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(body)
            });

            if (!response.ok) {
                const errorData = await response.json();
                console.error("Lỗi từ OpenRouteService:", errorData);
                throw new Error(`Không thể lấy dữ liệu đường đi cho ${currentLocation.name} -> ${destination.name}. Chi tiết: ${errorData.error?.message || response.statusText}`);
            }

            const data = await response.json();
            if (!data.features || data.features.length === 0) {
                throw new Error(`Không có tuyến đường trả về từ OpenRouteService cho ${currentLocation.name} -> ${destination.name}.`);
            }
            const color = colors[i % colors.length];
            const distance = (data.features[0].properties.summary.distance / 1000).toFixed(2);
            const duration = (data.features[0].properties.summary.duration / 60).toFixed(1);

            const routeLayer = L.geoJSON(data, {
                style: { color: color, weight: 5 },
                onEachFeature: (feature, layer) => {
                    layer.bindPopup(`
                        <b>Tuyến: ${segment.from} → ${segment.to}</b><br>
                        📏 Quãng đường: ${distance} km<br>
                        ⏱️ Thời gian dự kiến: ${duration} phút
                    `);
                }
            }).addTo(map);
            routeLayers.push(routeLayer);
            routeLayer.eachLayer((layer) => bounds.extend(layer.getBounds()));

            const coordinates = data.features[0].geometry.coordinates;
            const midpoint = getMidpoint(coordinates);
            const orderNumber = i + 1;
            const orderMarker = L.marker([midpoint[0], midpoint[1]], {
                icon: L.divIcon({
                    className: 'custom-div-icon',
                    html: `<div style="background-color: ${color}; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold;">${orderNumber}</div>`,
                    iconSize: [24, 24],
                    iconAnchor: [12, 12]
                })
            }).addTo(map);
            markers.push(orderMarker);

            if (!isReturnToStart) {
                const destMarker = L.marker([destination.lat, destination.lng], { icon: redIcon })
                    .addTo(map)
                    .bindPopup(`<b>${destination.name}</b>`);
                markers.push(destMarker);
            }

            const legInfo = document.createElement("div");
            legInfo.innerHTML = `
                <p><b>Chặng ${orderNumber}: ${segment.from} → ${segment.to}</b></p>
                <p>📏 Quãng đường: ${distance} km</p>
                <p>⏱️ Thời gian dự kiến: ${duration} phút</p>
                <hr>
            `;
            routeInfo.appendChild(legInfo);

            currentLocation = destination;
        }

        if (bounds.isValid()) {
            map.fitBounds(bounds, { padding: [50, 50] });
        } else if (markers.length > 0) {
            map.setView(markers[0].getLatLng(), 13);
        }
    } catch (error) {
        console.error("Có lỗi xảy ra khi hiển thị đường đi:", error);
        errorMessage.textContent = `Có lỗi xảy ra: ${error.message}`;
        errorMessage.style.backgroundColor = '#f8d7da';
        errorMessage.style.color = '#721c24';
        errorMessage.style.display = 'block';
        routeInfo.style.display = 'none';
    }
});

// Toggle Route Sidebar
document.getElementById('routeButton').addEventListener('click', () => {
    const sidebar = document.getElementById('routeSidebar');
    sidebar.classList.toggle('active');
});

// Initialize UI
document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM fully loaded and parsed. Running initial UI updates.");
    updateStoreSearch();
    updateVehicleOptions();
});