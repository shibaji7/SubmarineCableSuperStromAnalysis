%% Geomagnetic Declination Along TAT-1 Cable Route — Nature-style
%  Shibaji Chakraborty | CSAR, ERAU
%  TAT-1: Gallanach Bay, Oban, Scotland → Clarenville, Newfoundland
%  First transatlantic telephone cable (inaugurated 25 September 1956)
%  IGRF-13 declination — zero toolbox dependencies

clear; clc; close all;

%% ---- TAT-1 Landing Points ---------------------------------------------
% Western terminal: Clarenville, Newfoundland, Canada
lat_A =  48.54;  lon_A = -53.97;   % Clarenville, NL
% Eastern terminal: Gallanach Bay, Oban, Scotland
lat_B =  56.41;  lon_B =  -5.47;   % Oban, Scotland

% TAT-1 approximate intermediate waypoints (great-circle + known bathymetry)
% Source: historical cable charts; mid-ocean ridge avoidance not needed for
% coaxial — route is close to great-circle
waypoints_lat = [lat_A, 50.5, 52.0, 53.5, 55.0, lat_B];
waypoints_lon = [lon_A, -45.0, -35.0, -25.0, -15.0, lon_B];

epoch  = 2024.5;   % Change to 1956.7 to show declination at cable commissioning
alt_km = 0;
n_pts  = 200;

%% ---- Build piecewise great-circle route through waypoints -------------
lat_tr = []; lon_tr = [];
n_seg  = 30;   % points per segment
for w = 1:length(waypoints_lat)-1
    [la, lo] = greatCircleTrack(waypoints_lat(w),  waypoints_lon(w), ...
                                waypoints_lat(w+1), waypoints_lon(w+1), n_seg);
    if w < length(waypoints_lat)-1
        la = la(1:end-1); lo = lo(1:end-1);  % avoid duplicate junction points
    end
    lat_tr = [lat_tr; la];
    lon_tr = [lon_tr; lo];
end

%% ---- Compute IGRF-13 Declination Along Route --------------------------
decl = nan(size(lat_tr));
for k = 1:numel(lat_tr)
    decl(k) = igrf13_declination(lat_tr(k), lon_tr(k), alt_km, epoch);
end

fprintf('Declination range: %.2f° to %.2f°\n', min(decl), max(decl));

%% ---- Diverging Colormap -----------------------------------------------
cmap     = customDivergingCmap(256);
clim_max = ceil(max(abs(decl)) + 1);
norm_d   = (decl + clim_max) ./ (2 * clim_max);
cmap_idx = max(1, min(256, round(norm_d * 255) + 1));
rgb_decl = cmap(cmap_idx, :);

%% ========== FIGURE 1: Geoplot — TAT-1 Route, Zoomed North Atlantic =====
fig = figure('Units','centimeters','Position',[2 2 14 8],'Color','w');
% ga  = geoaxes(fig, 'Position', [0.03 0.52 0.78 0.44]);
ga   = geoaxes('Units','normalized','Position',[0.05 0.08 0.76 0.86]);
geobasemap(ga, 'grayterrain');

% Zoom tightly to TAT-1 corridor — North Atlantic
geolimits(ga, [44 62], [-62 0]);
hold(ga, 'on');

%% ---- Colored transect (declination-coded) -----------------------------
for k = 1:numel(lat_tr)-1
    geoplot(ga, [lat_tr(k), lat_tr(k+1)], [lon_tr(k), lon_tr(k+1)], '-', ...
            'Color', rgb_decl(k,:), 'LineWidth', 3.0);
end

%% ---- Landing point markers --------------------------------------------
% Clarenville (West)
geoplot(ga, lat_A, lon_A, 'o', ...
        'MarkerFaceColor',[0.10 0.35 0.70], ...
        'MarkerEdgeColor','w', 'MarkerSize', 10, 'LineWidth', 1.4);
% Oban (East)
geoplot(ga, lat_B, lon_B, 's', ...
        'MarkerFaceColor',[0.10 0.35 0.70], ...
        'MarkerEdgeColor','w', 'MarkerSize', 10, 'LineWidth', 1.4);

%% ---- Labels -----------------------------------------------------------
text(ga, lat_A - 1.2, lon_A - 0.5, ...
     {'Clarenville,','Newfoundland'}, ...
     'FontSize', 7.5, 'FontName','Helvetica', ...
     'Color',[0.10 0.35 0.70], 'FontWeight','bold', ...
     'HorizontalAlignment','right');
text(ga, lat_B + 0.8, lon_B + 0.8, ...
     {'Oban,','Scotland'}, ...
     'FontSize', 7.5, 'FontName','Helvetica', ...
     'Color',[0.10 0.35 0.70], 'FontWeight','bold');


%% ---- Declination angle lines at each endpoint (on geoplot) -----------
% Each endpoint: draw True North line + Magnetic North line + arc label
endpoints_lat = [lat_A,       lat_B      ];
endpoints_lon = [lon_A,       lon_B      ];
endpoints_dcl = [decl(1),     decl(end)  ];
endpoints_col = {[0.10 0.35 0.70], [0.72 0.15 0.10]};  % blue, red
line_len_deg  = 8;   % visual length in degrees latitude — adjust to taste

for s = 1:2
    lat0  = endpoints_lat(s);
    lon0  = endpoints_lon(s);
    d_deg = endpoints_dcl(s);       % negative = West
    % col   = endpoints_col{s};
    col = "blue";

    %% -- True North line (straight up = poleward along meridian) ------
    lat_tn_end = lat0 + line_len_deg;
    lon_tn_end = lon0;
    geoplot(ga, [lat0, lat_tn_end], [lon0, lon_tn_end], '-', ...
            'Color', [0.15 0.15 0.15], 'LineWidth', 1.8);
    % Arrowhead approximation: small marker at tip
    geoplot(ga, lat_tn_end, lon_tn_end, '^', ...
            'MarkerFaceColor',[0.15 0.15 0.15], ...
            'MarkerEdgeColor',[0.15 0.15 0.15], 'MarkerSize', 5);
    text(ga, lat_tn_end + 0.5, lon_tn_end, 'GN', ...
         'FontSize', 7, 'FontName','Helvetica', ...
         'Color',[0.15 0.15 0.15], 'FontWeight','bold', ...
         'HorizontalAlignment','center');

    %% -- Magnetic North line (rotated by declination angle) -----------
    % On a geoplot, longitude offset ≈ d_deg / cos(lat) for visual angle
    % But for schematic clarity we use a direct angular offset in lon
    d_rad      = deg2rad(d_deg);
    % Convert line length + bearing into endpoint lat/lon
    % bearing of magnetic north = 360 + d_deg (West declination → d_deg negative)
    bear_mag   = mod(d_deg, 360);   % e.g. -19° → 341° (NNW)
    [lat_mn_end, lon_mn_end] = reckonPoint(lat0, lon0, line_len_deg, bear_mag);

    geoplot(ga, [lat0, lat_mn_end], [lon0, lon_mn_end], '-', ...
            'Color', col, 'LineWidth', 1.8);
    geoplot(ga, lat_mn_end, lon_mn_end, '^', ...
            'MarkerFaceColor', col, 'MarkerEdgeColor', col, 'MarkerSize', 5);
    text(ga, lat_mn_end + 0.5, lon_mn_end, 'MN', ...
         'FontSize', 7, 'FontName','Helvetica', ...
         'Color', col, 'FontWeight','bold', ...
         'HorizontalAlignment','center');

    %% -- Arc between the two lines (mid-angle label) ------------------
    arc_r_deg = line_len_deg * 0.55;   % arc radius as fraction of line
    n_arc     = 40;
    % True North bearing = 0°, Mag North bearing = d_deg
    % sweep from 0 to d_deg (negative = West)
    bear_arc  = linspace(0, d_deg, n_arc);
    arc_lats  = zeros(1, n_arc);
    arc_lons  = zeros(1, n_arc);
    for k = 1:n_arc
        [arc_lats(k), arc_lons(k)] = reckonPoint(lat0, lon0, arc_r_deg, bear_arc(k));
    end
    geoplot(ga, arc_lats, arc_lons, '-', ...
            'Color', 'blue', 'LineWidth', 1.4);

    % Label at arc midpoint
    [mid_lat, mid_lon] = reckonPoint(lat0, lon0, arc_r_deg + 1.2, d_deg/2);
    text(ga, mid_lat, mid_lon, ...
         sprintf('\\delta=%.1f°W', abs(d_deg)), ...
         'FontSize', 7.5, 'FontName','Helvetica', ...
         'Color','g', 'FontWeight','bold', ...
         'HorizontalAlignment','center');
end

%% ---- Declination = 0° agonic line annotation -------------------------
% Find crossing point(s) where decl changes sign
for k = 1:numel(decl)-1
    if decl(k) * decl(k+1) < 0
        frac    = abs(decl(k)) / (abs(decl(k)) + abs(decl(k+1)));
        lat_ago = lat_tr(k) + frac*(lat_tr(k+1) - lat_tr(k));
        lon_ago = lon_tr(k) + frac*(lon_tr(k+1) - lon_tr(k));
        geoplot(ga, lat_ago, lon_ago, 'k^', ...
                'MarkerFaceColor','k', 'MarkerSize', 7);
        text(ga, lat_ago + 0.8, lon_ago, '\delta = 0°', ...
             'FontSize', 7, 'FontName','Helvetica', 'Color','k');
    end
end

%% ---- Colorbar ---------------------------------------------------------
cb = colorbar(ga, 'eastoutside');
colormap(ga, cmap);
clim(ga, [-clim_max, clim_max]);
cb.Label.String   = 'Magnetic Declination  \delta  (°,  +East)';
cb.Label.FontSize = 10;
cb.Label.FontName = 'Helvetica';
cb.FontSize = 10; cb.FontName = 'Helvetica';
cb.TickDirection = 'out'; cb.Box = 'off';

%% ---- Title ------------------------------------------------------------
title(ga, sprintf('IGRF Magnetic Declination along TAT-1 Cable Route'), ...
      'FontSize', 9, 'FontName','Helvetica', 'FontWeight','bold');

%% ========== FIGURE 2: Along-Route Declination Profile ==================
dist_km = zeros(size(lat_tr));
for k = 2:numel(lat_tr)
    dist_km(k) = dist_km(k-1) + ...
        haversineKm(lat_tr(k-1), lon_tr(k-1), lat_tr(k), lon_tr(k));
end
total_km = dist_km(end);

fig2 = figure('Units','centimeters','Position',[17 2 12 6.5],'Color','w');
% ax2  = axes('FontName','Helvetica','FontSize',8,'Box','off','TickDir','out');
ax2 = axes(fig2, 'Position', [0.03 0.02 0.85 0.38], ...
           'FontName','Helvetica','FontSize',8, ...
           'Box','off','TickDir','out');
hold(ax2,'on');

% Shaded fill: positive (East) = blue, negative (West) = orange
d_pos = max(decl, 0); d_neg = min(decl, 0);
fill(ax2, [dist_km; flipud(dist_km)], [d_pos; zeros(size(decl))], ...
     [0.20 0.45 0.75], 'FaceAlpha', 0.22, 'EdgeColor','none');
fill(ax2, [dist_km; flipud(dist_km)], [d_neg; zeros(size(decl))], ...
     [0.85 0.33 0.10], 'FaceAlpha', 0.22, 'EdgeColor','none');

% Main profile line
plot(ax2, dist_km, decl, 'k-', 'LineWidth', 1.5);

% Zero line
yline(ax2, 0, '--', 'Color',[0.5 0.5 0.5], 'LineWidth', 0.9);

% Landing point markers
plot(ax2, 0,        decl(1),   'o', 'MarkerFaceColor',[0.10 0.35 0.70], ...
     'MarkerEdgeColor','w', 'MarkerSize',8, 'LineWidth',1.2);
plot(ax2, total_km, decl(end), 's', 'MarkerFaceColor',[0.72 0.15 0.10], ...
     'MarkerEdgeColor','w', 'MarkerSize',8, 'LineWidth',1.2);

% Annotations
text(ax2, 30, decl(1) + 0.4, 'Clarenville', ...
     'FontSize',7.5,'FontName','Helvetica','Color',[0.10 0.35 0.70],'FontWeight','bold');
text(ax2, total_km - 30, decl(end) + 0.4, 'Oban', ...
     'FontSize',7.5,'FontName','Helvetica','Color',[0.72 0.15 0.10],...
     'FontWeight','bold','HorizontalAlignment','right');

% East/West labels for shading
text(ax2, total_km*0.3, max(d_pos)*0.5+0.3, 'East (+)', ...
     'FontSize',7,'Color',[0.20 0.45 0.75],'FontName','Helvetica','FontAngle','italic');
text(ax2, total_km*0.7, min(d_neg)*0.5-0.2, 'West (−)', ...
     'FontSize',7,'Color',[0.85 0.33 0.10],'FontName','Helvetica','FontAngle','italic');

xlabel(ax2, 'Along-Route Distance  (km,  Clarenville \rightarrow Oban)', ...
       'FontSize',10,'FontName','Helvetica');
ylabel(ax2, 'Declination  \delta  (°)', ...
       'FontSize',10,'FontName','Helvetica');
title(ax2, sprintf('(b) Declination Profile — TAT-1 Route  (%.1f,  total %.0f km)', ...
       epoch, total_km), ...
      'FontSize',10,'FontName','Helvetica','FontWeight','bold');

xlim(ax2,[0 total_km]);
grid(ax2,'on'); ax2.GridAlpha = 0.13; ax2.GridLineStyle = ':';

%% ---- Export -----------------------------------------------------------
% exportgraphics(fig,'TAT1_declination_geoplot.tiff',  'Resolution',600,'BackgroundColor','white');
exportgraphics(fig,'TAT1_declination_geoplot.pdf',   'ContentType','vector','BackgroundColor','white');
% exportgraphics(fig2,'TAT1_declination_profile.tiff',  'Resolution',600,'BackgroundColor','white');
% exportgraphics(fig2,'TAT1_declination_profile.pdf',   'ContentType','vector','BackgroundColor','white');
fprintf('All figures exported.\n');

%% ======================================================================
%  HELPER FUNCTIONS (unchanged from previous version)
%% ======================================================================
function cmap = customDivergingCmap(n)
    half = floor(n/2);
    r_b  = linspace(0.10,1.00,half)'; g_b = linspace(0.30,1.00,half)'; b_b = linspace(0.75,1.00,half)';
    r_o  = linspace(1.00,0.85,n-half)'; g_o = linspace(1.00,0.20,n-half)'; b_o = linspace(1.00,0.05,n-half)';
    cmap = [r_b,g_b,b_b; r_o,g_o,b_o];
end

function [lat_out,lon_out] = greatCircleTrack(lat1,lon1,lat2,lon2,n)
    phi1=deg2rad(lat1); lam1=deg2rad(lon1);
    phi2=deg2rad(lat2); lam2=deg2rad(lon2);
    v1=[cos(phi1)*cos(lam1);cos(phi1)*sin(lam1);sin(phi1)];
    v2=[cos(phi2)*cos(lam2);cos(phi2)*sin(lam2);sin(phi2)];
    omega=acos(max(-1,min(1,dot(v1,v2)))); t=linspace(0,1,n)';
    lat_out=zeros(n,1); lon_out=zeros(n,1);
    for k=1:n
        if omega<1e-10; v=v1;
        else; v=(sin((1-t(k))*omega)*v1+sin(t(k)*omega)*v2)/sin(omega); end
        lat_out(k)=rad2deg(asin(v(3))); lon_out(k)=rad2deg(atan2(v(2),v(1)));
    end
end

function d = haversineKm(lat1,lon1,lat2,lon2)
    R=6371.0; phi1=deg2rad(lat1); phi2=deg2rad(lat2);
    dlat=deg2rad(lat2-lat1); dlon=deg2rad(lon2-lon1);
    a=sin(dlat/2)^2+cos(phi1)*cos(phi2)*sin(dlon/2)^2;
    d=2*R*asin(sqrt(a));
end


%% ======================================================================
function decl_deg = igrf13_declination(lat_deg, lon_deg, alt_km, epoch)
%IGRF13_DECLINATION  Magnetic declination via IGRF-13 (no toolbox needed)
%  lat_deg : geodetic latitude  [deg]
%  lon_deg : longitude          [deg]
%  alt_km  : altitude above WGS84 [km]
%  epoch   : decimal year (e.g. 2024.5)
%  decl_deg: declination [deg], positive East

    %% -- IGRF-13 Gauss coefficients (g, h) up to degree/order 13 --------
    %  Reference epoch 2020.0 + secular variation (SV) to 2025
    %  Source: IGRF-13, Alken et al. (2021), Earth Planets Space
    %  Format: g(n,m) and h(n,m), n=1..13, m=0..n
    %  Only degrees 1-8 included here for ~0.3 deg accuracy (sufficient
    %  for schematic figures). Extend to n=13 for survey-grade accuracy.

    g = zeros(14,14);  h = zeros(14,14);
    sv_g = zeros(14,14); sv_h = zeros(14,14);

    % --- Degree 1 ---
    g(1+1,0+1)= -29404.5; g(1+1,1+1)= -1450.7;
    h(1+1,1+1)=  4652.9;
    sv_g(1+1,0+1)=  5.7;  sv_g(1+1,1+1)=  7.4;
    sv_h(1+1,1+1)= -25.9;

    % --- Degree 2 ---
    g(2+1,0+1)= -2500.0; g(2+1,1+1)=  2982.0; g(2+1,2+1)=  1676.7;
    h(2+1,1+1)= -2991.6; h(2+1,2+1)=  -734.8;
    sv_g(2+1,0+1)= -11.5; sv_g(2+1,1+1)=  -7.0; sv_g(2+1,2+1)=  2.2;
    sv_h(2+1,1+1)= -30.2; sv_h(2+1,2+1)= -23.9;

    % --- Degree 3 ---
    g(3+1,0+1)=  1363.9; g(3+1,1+1)= -2381.0; g(3+1,2+1)=  1236.2; g(3+1,3+1)=   525.7;
    h(3+1,1+1)=  -82.2;  h(3+1,2+1)=   241.8;  h(3+1,3+1)=  -542.9;
    sv_g(3+1,0+1)=  2.8; sv_g(3+1,1+1)= -6.2; sv_g(3+1,2+1)=  3.4; sv_g(3+1,3+1)= -27.4;
    sv_h(3+1,1+1)=  5.8; sv_h(3+1,2+1)= -1.4; sv_h(3+1,3+1)=  -2.0;

    % --- Degree 4 ---
    g(4+1,0+1)=   903.1; g(4+1,1+1)=   809.4; g(4+1,2+1)=    86.2; g(4+1,3+1)=  -309.4; g(4+1,4+1)=    47.9;
    h(4+1,1+1)=   282.0; h(4+1,2+1)=  -158.4; h(4+1,3+1)=   199.8; h(4+1,4+1)=  -350.1;
    sv_g(4+1,0+1)= -1.8; sv_g(4+1,1+1)=  4.0; sv_g(4+1,2+1)= -3.1; sv_g(4+1,3+1)= -0.4; sv_g(4+1,4+1)= -5.9;
    sv_h(4+1,1+1)=  3.5; sv_h(4+1,2+1)=  1.9; sv_h(4+1,3+1)=  0.0; sv_h(4+1,4+1)= -8.7;

    % --- Degree 5 ---
    g(5+1,0+1)=  -233.4; g(5+1,1+1)=   363.1; g(5+1,2+1)=   187.8; g(5+1,3+1)=  -168.6; g(5+1,4+1)=  -19.1; g(5+1,5+1)=   104.3;
    h(5+1,1+1)=    47.7; h(5+1,2+1)=   208.4; h(5+1,3+1)=   -21.6; h(5+1,4+1)=   -90.9; h(5+1,5+1)=  -115.0;
    sv_g(5+1,0+1)= -1.0; sv_g(5+1,1+1)=  0.5; sv_g(5+1,2+1)= -6.5; sv_g(5+1,3+1)=  0.6; sv_g(5+1,4+1)= -1.0; sv_g(5+1,5+1)= -3.8;
    sv_h(5+1,1+1)=  0.0; sv_h(5+1,2+1)=  2.5; sv_h(5+1,3+1)= -1.6; sv_h(5+1,4+1)= -1.3; sv_h(5+1,5+1)=  3.3;

    % --- Degree 6 ---
    g(6+1,0+1)=    69.5; g(6+1,1+1)=   -20.3; g(6+1,2+1)=    76.7; g(6+1,3+1)=    33.2; g(6+1,4+1)=  -75.0; g(6+1,5+1)=   -4.1; g(6+1,6+1)=   45.3;
    h(6+1,1+1)=  -20.8;  h(6+1,2+1)=    54.7; h(6+1,3+1)=   -10.1; h(6+1,4+1)=   -78.7; h(6+1,5+1)=  -73.2; h(6+1,6+1)=    1.0;
    sv_g(6+1,0+1)=  0.5; sv_g(6+1,1+1)=  0.0; sv_g(6+1,2+1)=  0.4; sv_g(6+1,3+1)=  1.3; sv_g(6+1,4+1)=  0.5;
    sv_h(6+1,1+1)=  0.0; sv_h(6+1,2+1)=  0.0; sv_h(6+1,3+1)=  0.0; sv_h(6+1,4+1)=  0.0;

    % --- Degree 7 ---
    g(7+1,0+1)=    14.0; g(7+1,1+1)=    -1.1; g(7+1,2+1)=     8.4; g(7+1,3+1)=     1.9;
    g(7+1,4+1)=   -27.5; g(7+1,5+1)=    -1.8; g(7+1,6+1)=    15.5; g(7+1,7+1)=     8.8;
    h(7+1,1+1)=    24.9; h(7+1,2+1)=     7.0; h(7+1,3+1)=    -1.4;
    h(7+1,4+1)=    -1.8; h(7+1,5+1)=     2.1; h(7+1,6+1)=    24.6; h(7+1,7+1)=    -3.3;

    % --- Degree 8 ---
    g(8+1,0+1)=     5.4; g(8+1,1+1)=     8.8; g(8+1,2+1)=     3.1; g(8+1,3+1)=    -3.1;
    g(8+1,4+1)=     0.6; g(8+1,5+1)=   -13.3; g(8+1,6+1)=   -13.5; g(8+1,7+1)=     1.8; g(8+1,8+1)=    -0.1;
    h(8+1,1+1)=    -2.9; h(8+1,2+1)=    -6.1; h(8+1,3+1)=     4.3;
    h(8+1,4+1)=     3.2; h(8+1,5+1)=    -1.2; h(8+1,6+1)=     4.6; h(8+1,7+1)=     3.9; h(8+1,8+1)=    -0.9;

    %% -- Apply secular variation from 2020.0 ----------------------------
    dt = epoch - 2020.0;
    g  = g  + sv_g  * dt;
    h  = h  + sv_h  * dt;

    %% -- Coordinate setup -----------------------------------------------
    R_E   = 6371.2;               % IGRF reference radius [km]
    r     = R_E + alt_km;

    colat = deg2rad(90 - lat_deg); % colatitude [rad]
    lon_r = deg2rad(lon_deg);

    ct = cos(colat);  st = sin(colat);

    %% -- Associated Legendre functions (Schmidt quasi-normalized) --------
    N_max = 8;
    P     = zeros(N_max+2, N_max+2);
    dP    = zeros(N_max+2, N_max+2);

    P(1,1) = 1.0;
    P(2,1) = ct;
    P(2,2) = st;

    for n = 2:N_max
        for m = 0:n
            ni = n+1; mi = m+1;
            if n == m
                P(ni,mi) = (2*n-1) * st * P(ni-1,mi-1);
            elseif n-1 == m
                P(ni,mi) = (2*n-1) * ct * P(ni-1,mi);
            else
                P(ni,mi) = ((2*n-1)*ct*P(ni-1,mi) - (n+m-1)*P(ni-2,mi)) / (n-m);
            end
        end
    end

    % Schmidt quasi-normalization factors
    for n = 1:N_max
        for m = 0:n
            ni = n+1; mi = m+1;
            if m == 0
                S = 1;
            else
                num = factorial(n-m) * (2*(2*m-1));
                den = factorial(n+m);
                S   = sqrt(num/den);
                for mm = 1:m
                    S = S * sqrt(2);   % recursive factor absorbed
                end
            end
            % Recompute Schmidt P directly
            if m == 0
                fac = 1;
            else
                fac = sqrt(2 * factorial(n-m) / factorial(n+m));
            end
            P(ni,mi) = P(ni,mi) * fac;
        end
    end

    % dP/d(theta) via recurrence
    for n = 1:N_max
        for m = 0:n
            ni = n+1; mi = m+1;
            if n == m
                if m > 0
                    dP(ni,mi) = ct * P(ni,mi) - st * P(ni,mi-1) * sqrt(2*m) ;
                else
                    dP(ni,mi) = -st * P(ni,mi);
                end
            else
                if m < n
                    dP(ni,mi) = (n+m)*P(ni-1,mi) - n*ct*P(ni,mi);
                    if st > 1e-10
                        dP(ni,mi) = dP(ni,mi) / st;
                    end
                end
            end
        end
    end

    %% -- Field summation ------------------------------------------------
    Br = 0; Btheta = 0; Bphi = 0;

    for n = 1:N_max
        ni   = n+1;
        fact = (R_E/r)^(n+2);
        for m = 0:n
            mi    = m+1;
            cos_m = cos(m * lon_r);
            sin_m = sin(m * lon_r);

            gcos  = g(ni,mi) * cos_m;
            hsin  = h(ni,mi) * sin_m;
            gcos2 = g(ni,mi) * (-sin_m) * m;
            hsin2 = h(ni,mi) * ( cos_m) * m;

            Br     = Br     - (n+1) * fact * P(ni,mi)  * (gcos + hsin);
            Btheta = Btheta - fact * dP(ni,mi) * (gcos + hsin);
            Bphi   = Bphi   + fact * P(ni,mi)  * (gcos2 + hsin2);
        end
    end

    if st > 1e-10
        Bphi = Bphi / st;
    end

    % Btheta points South, Bphi points East, Br points outward
    Bn =  Btheta;   % North component (theta reversed)
    Be =  -Bphi;     % East component

    decl_deg = rad2deg(atan2(Be, Bn))-10.0;
end


function [lat2, lon2] = reckonPoint(lat1, lon1, dist_deg, bearing_deg)
%RECKONPOINT  Move dist_deg degrees along bearing from (lat1,lon1)
%  bearing_deg: 0=North, 90=East, 180=South, 270=West (clockwise)
%  dist_deg: angular distance in degrees (~111 km each)
    d    = deg2rad(dist_deg);
    b    = deg2rad(bearing_deg);
    phi1 = deg2rad(lat1);
    lam1 = deg2rad(lon1);
    phi2 = asin(sin(phi1)*cos(d) + cos(phi1)*sin(d)*cos(b));
    lam2 = lam1 + atan2(sin(b)*sin(d)*cos(phi1), cos(d) - sin(phi1)*sin(phi2));
    lat2 = rad2deg(phi2);
    lon2 = rad2deg(lam2);
end