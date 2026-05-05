%% 🌌 Earth with Dipole Field Lines & Station Markers
clc; clear; close all;

% matlab -batch "addpath('py/tat1'); plot_figure6"

function earth_texture = get_earth_texture()

    out_file = fullfile('data/earth.jpg');

    % If already exists → load
    if isfile(out_file)
        earth_texture = imread(out_file);
        return;
    end

    fprintf('Downloading Earth texture...\n');

    % Multiple reliable sources (fallbacks)
    urls = {
        'https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57730/land_ocean_ice_2048.jpg', ...
        'https://upload.wikimedia.org/wikipedia/commons/2/2c/Blue_Marble_2002.png', ...
        'https://upload.wikimedia.org/wikipedia/commons/6/6f/Earth_Eastern_Hemisphere.jpg'
    };

    success = false;

    for i = 1:length(urls)
        try
            websave(out_file, urls{i});
            fprintf('Downloaded from source %d\n', i);
            success = true;
            break;
        catch
            fprintf('Source %d failed...\n', i);
        end
    end

    if ~success
        error('All Earth texture download sources failed.');
    end

    earth_texture = imread(out_file);

end

%% 🌎 Create Earth
[x, y, z] = sphere(300);

% --- Axial tilt toward Sun (23.5°, realistic obliquity) ---
sun_lon_deg = 120; % East China (Shanghai / Nanjing region)
tilt_angle  = deg2rad(23.5);
sun_lon_rad = deg2rad(sun_lon_deg);
ux = -sin(sun_lon_rad);   % rotation axis = Z × Sun_dir
uy =  cos(sun_lon_rad);
c = cos(tilt_angle); s = sin(tilt_angle); t = 1 - c;
R_tilt = [t*ux^2+c,  t*ux*uy,   s*uy; ...
          t*ux*uy,   t*uy^2+c, -s*ux; ...
         -s*uy,       s*ux,      c  ];
pts = R_tilt * [x(:)'; y(:)'; z(:)'];
x = reshape(pts(1,:), size(x));
y = reshape(pts(2,:), size(y));
z = reshape(pts(3,:), size(z));
% --- End tilt ---

figure('Color','k'); % black background (space)
hold on;

% Load Earth texture (download NASA Blue Marble as 'earth.jpg')
earth_texture = get_earth_texture();

surf(x, y, z, flipud(earth_texture), ...
    'FaceColor','texturemap', ...
    'EdgeColor','none');

axis equal off;

%% 💡 Lighting - matches Sun position (East China/120°E)
% sun_lon_deg already set above (120); reuse it here
sun_lon_rad = deg2rad(sun_lon_deg);
light_pos = [cos(sun_lon_rad), sin(sun_lon_rad), 0];
light('Position', light_pos, 'Style', 'infinite');
lighting phong;
material dull;

%% 🧲 Dipole-like magnetic field lines (approx IGRF)
Re = 1; % Earth radius

L_values = [2, 4, 6, 8]; % L-shells
theta = linspace(-pi/2, pi/2, 200); % latitude-like angle

% Multiple longitude offsets for global coverage
lon_offsets = linspace(0, 2*pi, 5); % 4 evenly spaced longitudes
lon_offsets(end) = []; % remove 360° (redundant)

for lon_offset = lon_offsets
    for L = L_values
        r = L * (cos(theta)).^2; % dipole equation
        
        % Meridian plane coordinates
        x_m = r .* cos(theta);
        z_m = r .* sin(theta);
        y_m = zeros(size(x_m));
        
        % Rotate to longitude
        x_rot = x_m * cos(lon_offset) - y_m * sin(lon_offset);
        y_rot = x_m * sin(lon_offset) + y_m * cos(lon_offset);
        z_rot = z_m;
        
        plot3(x_rot, y_rot, z_rot, 'm', 'LineWidth', 0.4);
    end
end

%% 🌡️ Magnetic latitude circles (dipole field)
% Rotate geographic coordinates to magnetic coordinate system
% Magnetic pole at ~80°N, 72°W (~-72°)
mag_pole_lat = deg2rad(80);
mag_pole_lon = deg2rad(-72);

mag_lats = [-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75]; % magnetic latitudes

for mlat_deg = mag_lats
    % Geographic latitude that corresponds to this magnetic latitude
    % Using simple offset: geographic lat = magnetic lat + offset
    % The offset varies with longitude, but roughly -11° near magnetic pole
    
    nlon = 100;
    lon_vals = linspace(0, 2*pi, nlon);
    
    X_arr = []; Y_arr = []; Z_arr = [];
    
    for j = 1:nlon
        lon = lon_vals(j);
        
        % Simple approximation: shift geographic lat based on proximity to magnetic pole
        lambda = lon + mag_pole_lon;
        
        % At magnetic equator (lon = mag_pole_lon), offset is ~11°
        % At poles, offset is 0
        offset = deg2rad(11) * cos(lambda);
        
        glat = deg2rad(mlat_deg) + offset;
        
        X = cos(glat) * cos(lon);
        Y = cos(glat) * sin(lon);
        Z = sin(glat);
        
        X_arr = [X_arr, X];
        Y_arr = [Y_arr, Y];
        Z_arr = [Z_arr, Z];
    end
    
    plot3(X_arr, Y_arr, Z_arr, 'w--', 'LineWidth', 0.5);
end

%% ☀️ Geocentric Solar Magnetospheric (GSM) reference
% GSM X-axis points toward Sun
% Sun over East China: ~120°E
sun_lon = deg2rad(120); % Sun over East China (Shanghai region)
sun_lat = 0;

% Sun direction vector (X-GSM)
X_sun = cos(sun_lat)*cos(sun_lon);
Y_sun = cos(sun_lat)*sin(sun_lon);
Z_sun = sin(sun_lat);

% Draw Sun-Earth line (X-GSM) - show both directions
% Night side (negative, facing Atlantic viewer) drawn solid
% Sun side (positive, behind Earth from this view) drawn dashed
t_sun_night = linspace(-1.8, -1, 80);
t_sun_day   = linspace(-1, 9.0, 80);
plot3(t_sun_night*X_sun, t_sun_night*Y_sun, t_sun_night*Z_sun, ...
    'Color',[1 0.8 0], 'LineWidth', 0.5);
plot3(t_sun_day*X_sun, t_sun_day*Y_sun, t_sun_day*Z_sun, ...
    'Color',[1 0.8 0], 'LineWidth', 0.5, 'LineStyle','-');
hold on;
% Arrowhead on night side (toward viewer)
plot3(-1.6*X_sun, -1.6*Y_sun, -1.6*Z_sun, 'v', ...
    'Color',[1 0.8 0], 'MarkerSize', 4, 'MarkerFaceColor',[1 0.8 0]);

% GSM Z-axis (magnetic dipole axis, northward)
mag_pole_lat = deg2rad(80);
mag_pole_lon = deg2rad(-72);
X_mag = cos(mag_pole_lat)*cos(mag_pole_lon);
Y_mag = cos(mag_pole_lat)*sin(mag_pole_lon);
Z_mag = sin(mag_pole_lat);

% Draw magnetic axis (Z-GSM)
t_mag = linspace(-1.5, 1.5, 100);
plot3(t_mag*X_mag, t_mag*Y_mag, t_mag*Z_mag, 'g-', 'LineWidth', 0.5);
plot3(X_mag, Y_mag, Z_mag, 'g.', 'MarkerSize', 10);

% GSM Y-axis (dawn-dusk) — normalize the direction vector explicitly
% (-Y_mag, X_mag, 0) is correct direction but NOT unit-length; norm ≈ 0.17
Y_gsm_raw = [-Y_mag, X_mag, 0];
Y_gsm_nrm = norm(Y_gsm_raw);
X_gsm_y = Y_gsm_raw(1) / Y_gsm_nrm;  % normalized
Y_gsm_y = Y_gsm_raw(2) / Y_gsm_nrm;
Z_gsm_y = 0;

t_y = linspace(-1.8, 1.8, 100);
plot3(t_y*X_gsm_y, t_y*Y_gsm_y, t_y*Z_gsm_y, 'Color',[1 0.2 0.2], 'LineWidth', 0.5);
% Arrowheads on both ends
plot3( 1.6*X_gsm_y,  1.6*Y_gsm_y,  0, 'o', 'Color',[1 0.2 0.2], 'MarkerSize', 4, 'MarkerFaceColor',[1 0.2 0.2]);
plot3(-1.6*X_gsm_y, -1.6*Y_gsm_y,  0, 'o', 'Color',[1 0.2 0.2], 'MarkerSize', 4, 'MarkerFaceColor',[1 0.2 0.2]);

% GSM labels — smaller font
% X_GSM: Sun side label raised in Z so it clears the Earth disk in projection
text( 2.0*X_sun,  2.0*Y_sun,  0.6, 'X_{GSM}(Sun)', 'Color','y', 'FontSize', 3.5);
text(-1.9*X_sun, -1.9*Y_sun,  0.1, '-X_{GSM}',     'Color','y', 'FontSize', 3);
text(1.6*X_mag, 1.6*Y_mag, 1.6*Z_mag, 'Z_{GSM}', 'Color','g', 'FontSize', 3.5);
text( 1.9*X_gsm_y,  1.9*Y_gsm_y,  0.15, 'Y_{GSM}',  'Color',[1 0.2 0.2], 'FontSize', 3.5);
text(-1.9*X_gsm_y, -1.9*Y_gsm_y,  0.15, '-Y_{GSM}', 'Color',[1 0.2 0.2], 'FontSize', 3);

%% 📍 Station locations (FRD & ESK)
% FRD: Fredericksburg (~38.2N, -77.4E) - 11 UT (dawn/early morning)
% ESK: Eskdalemuir (~55.3N, -3.2E) - 2 UT (night)

stations = struct( ...
    'name', {'FRD','ESK'}, ...
    'lat',  {38.2, 55.3}, ...
    'lon',  {-77.4, -3.2}, ...
    'text_lat', {-5, -10}, ...
    'text_lon', {-10, 3}); % UT time for each station

% Submarine cable terminals
% Western: Clarenville, Newfoundland, Canada
lat_A = 48.54; lon_A = -53.97;
% Eastern: Gallanach Bay, Oban, Scotland
lat_B = 56.41; lon_B = -5.47;



for i = 1:length(stations)
    lat = deg2rad(stations(i).lat-.5);
    lon = deg2rad(stations(i).lon-.5);
    
    X = cos(lat).*cos(lon);
    Y = cos(lat).*sin(lon);
    Z = sin(lat);
    
    % Plot station marker slightly above surface
    R = 1.00;
    plot3(R*X, R*Y, 1.08*Z, 'rs', 'MarkerSize', 3, 'MarkerFaceColor','r', 'LineWidth', 0.01);
    
    % Station label - adjusted position
    text_lat = stations(i).lat + stations(i).text_lat;
    text_lon = stations(i).lon + stations(i).text_lon;
    X = cos(deg2rad(text_lat)) * cos(deg2rad(text_lon));
    Y = cos(deg2rad(text_lat)) * sin(deg2rad(text_lon));
    label_str = sprintf('%s', stations(i).name);
    text(X, Y, 1.2*Z, label_str, ...
        'Color','r', 'FontSize', 4, 'FontWeight','bold');
end

%% ⚡ Auroral electrojet arrows — curved arcs along MLAT circles
% Uses same MLAT approximation as the dashed latitude circles above:
%   geographic_lat(rad) = mlat + 11deg * cos(glon_deg - 72)
R_ej         = 1.18;   % elevation above surface (Earth radii)
arc_span_deg = 28;     % half-span of arc in geographic longitude degrees
n_arc        = 80;

mlat_to_glat = @(mlat_deg, glon_deg) ...
    deg2rad(mlat_deg) + deg2rad(11) * cos(deg2rad(glon_deg - 72));

% Estimate MLAT of each station: mlat = glat - offset
esk_lon_deg = -3.2;  esk_lat_deg = 60.3;
frd_lon_deg = -77.4; frd_lat_deg = 35.2;
mlat_esk = esk_lat_deg - rad2deg(deg2rad(11) * cos(deg2rad(esk_lon_deg - 72)));
mlat_frd = frd_lat_deg - rad2deg(deg2rad(11) * cos(deg2rad(frd_lon_deg - 72)));

% --- ESK: Westward Electrojet — west end near Clarenville (Cl, ~-54°W) ---
lon_arc_esk = linspace(lon_A, esk_lon_deg + 20, n_arc);
Xa = zeros(1,n_arc); Ya = zeros(1,n_arc); Za = zeros(1,n_arc);
for k = 1:n_arc
    gl = mlat_to_glat(mlat_esk, lon_arc_esk(k));
    Xa(k) = R_ej * cos(gl) * cos(deg2rad(lon_arc_esk(k)));
    Ya(k) = R_ej * cos(gl) * sin(deg2rad(lon_arc_esk(k)));
    Za(k) = R_ej * sin(gl);
end
plot3(Xa, Ya, Za, 'Color',[0.4 0.8 1], 'LineWidth', 2.5);
d = [Xa(1)-Xa(3), Ya(1)-Ya(3), Za(1)-Za(3)];
d = d / norm(d) * 0.04;
quiver3(Xa(3), Ya(3), Za(3), d(1), d(2), d(3), ...
    0, 'Color',[0.4 0.8 1], 'LineWidth', 2.5, 'MaxHeadSize', 4);
text(Xa(1)-0.2, Ya(1)-0.01, Za(1)+0.05, ...
    'J_{w}', 'Color',[0.4 0.8 1], 'FontSize', 4, 'FontWeight','bold');

% --- FRD: Eastward Electrojet — east end near Clarenville (Cl, ~-54°W) ---
lon_arc_frd = linspace(frd_lon_deg - 14, lon_A, n_arc);
Xb = zeros(1,n_arc); Yb = zeros(1,n_arc); Zb = zeros(1,n_arc);
for k = 1:n_arc
    gl = mlat_to_glat(mlat_frd, lon_arc_frd(k));
    Xb(k) = R_ej * cos(gl) * cos(deg2rad(lon_arc_frd(k)));
    Yb(k) = R_ej * cos(gl) * sin(deg2rad(lon_arc_frd(k)));
    Zb(k) = R_ej * sin(gl);
end
plot3(Xb, Yb, Zb, 'Color',[1 0.5 0], 'LineWidth', 2.5);
d = [Xb(end)-Xb(end-2), Yb(end)-Yb(end-2), Zb(end)-Zb(end-2)];
d = d / norm(d) * 0.04;
quiver3(Xb(end-2), Yb(end-2), Zb(end-2), d(1), d(2), d(3), ...
    0, 'Color',[1 0.5 0], 'LineWidth', 2.5, 'MaxHeadSize', 4);
text(Xb(40)-0.03, Yb(40)+0.03, Zb(40)+0.05, ...
    'J_{e}', 'Color',[1 0.5 0], 'FontSize', 4, 'FontWeight','bold');

%% 🔗 Submarine cable route (TAT-1 waypoints — spline smoothed)
lat_cable_wp = [lat_A, 50.5, 52.0, 53.5, 55.0, lat_B];
lon_cable_wp = [lon_A, -45.0, -35.0, -25.0, -15.0, lon_B];

t_wp   = linspace(0, 1, numel(lat_cable_wp));
t_fine = linspace(0, 1, 300);
lat_cable = interp1(t_wp, lat_cable_wp, t_fine, 'spline');
lon_cable = interp1(t_wp, lon_cable_wp, t_fine, 'spline');

rlat_cable = deg2rad(lat_cable);
rlon_cable = deg2rad(lon_cable);

X_cable = cos(rlat_cable) .* cos(rlon_cable);
Y_cable = cos(rlat_cable) .* sin(rlon_cable);
Z_cable = sin(rlat_cable);
plot3(X_cable, Y_cable, Z_cable, 'W-', 'LineWidth', 3.0);

cable_ends = struct( ...
    'name', {'Cl','Ob'}, ...
    'lat',  {lat_cable(1), lat_cable(6)}, ...
    'lon',  {lon_cable(1), lon_cable(6)}, ...
    'text_lat', {-2, -15}, ...
    'text_lon', {-10, -15});

%% 📡 Submarine cable terminals
for i = 1:length(cable_ends)
    lat = deg2rad(cable_ends(i).lat);
    lon = deg2rad(cable_ends(i).lon);
    
    X = cos(lat).*cos(lon);
    Y = cos(lat).*sin(lon);
    Z = sin(lat);
    
    % Plot cable terminal marker — white filled circle, sized to match cable LineWidth
    R = 1.0;
    plot3(R*X, R*Y, R*Z, 'o', 'MarkerSize', 3, ...
        'MarkerFaceColor','w', 'MarkerEdgeColor','w', 'LineWidth', 0.5);
    
    % Cable label - adjusted position
    text_lat = cable_ends(i).lat + cable_ends(i).text_lat;
    text_lon = cable_ends(i).lon + cable_ends(i).text_lon;
    X = cos(deg2rad(text_lat)) * cos(deg2rad(text_lon));
    Y = cos(deg2rad(text_lat)) * sin(deg2rad(text_lon));
    label_str = sprintf('%s', cable_ends(i).name);
    text(X, Y, 1.08*Z, label_str, ...
        'Color','c', 'FontSize', 4, 'FontWeight','bold');
end

%% 🌙 View from night side (Atlantic-facing)
zoom(6); % zoom in for better visibility
view([30 30]); % Atlantic night-side view (facing Europe/Atlantic from space)

%% 📸 Save figure
set(gcf, 'InvertHardcopy', 'off'); % preserve dark background
print('figures/tat1/earth_fieldlines_nightside.png','-dpng','-r1000');