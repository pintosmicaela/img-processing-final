pkg load image;
%Detectar círculos y líneas en las fotos que tomaron previamente.
%Contar cuántas galletitas hay de cada tipo y color.

function img_filtered_gauss = apply_gauss_filter(img, sigma, kernel)
  gaussian_filter = fspecial('gaussian', kernel, sigma);
  img_filtered_gauss = imfilter(img, gaussian_filter, 'symmetric');
end

function img_edges = apply_canny(img, min_details, max_details)
  img_edges = edge(img, 'Canny', [min_details max_details]);
end

function img_filtered_med = apply_median_filter(img, kernel)
  mask_size = [kernel, kernel];
  img_filtered_med = medfilt2(img, mask_size);
end

function img_clean_edges = clean_edges(img, kernel)
  se = strel('square', kernel);
  img_dilatada = imdilate(img, se);
  img_clean_edges = imerode(img_dilatada, se);
end

function img_with_filters = apply_all_filter(img, kernel_med, sigma, kernel_gauss, min, max, kernel_clean)
  img_gray = rgb2gray(img);
  img_med = apply_median_filter(img_gray, kernel_med);
  img_gauss = apply_gauss_filter(img_med, sigma, kernel_gauss);
  img_canny = apply_canny(img_gauss, min, max);
  img_with_filters = clean_edges(img_canny, kernel_clean);
end


function draw_lines(img_reference, img_original, nombre_archivo)
  % Transformada de Hough
  [H, theta, rho] = hough(img_reference);
  % Detectar picos
  threshold = max(H(:)) * 0.3; % Ajusta según la calidad de bordes
  peaks = houghpeaks(H, 20, 'Threshold', threshold, 'NHoodSize', [21, 21]);
  % Dibujar líneas detectadas
  lines = houghlines(img_reference, theta, rho, peaks, 'FillGap', 5, 'MinLength', 30);
  imshow(img_original);
  hold on;
  for k = 1:length(lines)
      xy = [lines(k).point1; lines(k).point2];
      plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'red');
  end
  hold off;
  saveas(gcf, nombre_archivo);
end

function ellipse(a, b, theta, xc, yc, color)
    t = linspace(0, 2*pi, 100);
    x = a * cos(t);
    y = b * sin(t);

    R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
    coords = R * [x; y];
    x_rot = coords(1, :) + xc;
    y_rot = coords(2, :) + yc;

    plot(x_rot, y_rot, color, 'LineWidth', 1.5);
end

function ellipse_reference(img)
  [y, x] = find(img);
  % Calcular momentos
  m00 = length(x); % Momento de orden 0
  m10 = sum(x); % Momento de orden 1
  m01 = sum(y);
  x_bar = m10 / m00; % Centroide (x)
  y_bar = m01 / m00; % Centroide (y)
  % Momentos centrales
  mu20 = sum((x - x_bar).^2);
  mu02 = sum((y - y_bar).^2);
  mu11 = sum((x - x_bar) .* (y - y_bar));
  % Ejes de la elipse
  a = sqrt((mu20 + mu02 + sqrt((mu20 - mu02)^2 + 4*mu11^2)) / m00);
  b = sqrt((mu20 + mu02 - sqrt((mu20 - mu02)^2 + 4*mu11^2)) / m00);
  % Orientación
  theta = 0.5 * atan2(2*mu11, mu20 - mu02);
  ellipse_data = struct('xc', x_bar, 'yc', y_bar, 'a', a, 'b', b, 'theta', theta);
  save('ellipse_data.mat', 'ellipse_data');
end

function cant_elipsis = draw_elipses(img_reference, img, img_original, nombre_archivo)
  ellipse_reference(img_reference);
  [labels, num] = bwlabel(img);
  load('ellipse_data.mat');
  cant_elipsis = 0;
  imshow(img_original); hold on;

  for k = 1:num
      % Extraer la región actual
      [y, x] = find(labels == k);
      % Calcular momentos
      m00 = length(x); % Momento de orden 0 (área)
      m10 = sum(x); % Momento de orden 1
      m01 = sum(y);
      x_bar = m10 / m00; % Centroide (x)
      y_bar = m01 / m00; % Centroide (y)
      % Momentos centrales
      mu20 = sum((x - x_bar).^2);
      mu02 = sum((y - y_bar).^2);
      mu11 = sum((x - x_bar) .* (y - y_bar));
      % Calcular los ejes y orientación de la elipse
      a = sqrt((mu20 + mu02 + sqrt((mu20 - mu02)^2 + 4*mu11^2)) / m00);
      b = sqrt((mu20 + mu02 - sqrt((mu20 - mu02)^2 + 4*mu11^2)) / m00);
      theta = 0.5 * atan2(2*mu11, mu20 - mu02);

      diff_a = abs(ellipse_data.a - a);
      diff_b = abs(ellipse_data.b - b);

      min_area = ellipse_data.a * ellipse_data.b * pi;
      area = a * b * pi;

      if area >= min_area && diff_a < 200 && diff_b < 200
         ellipse(a, b, theta, x_bar, y_bar, 'r');
         cant_elipsis = cant_elipsis + 1;
      end
  end
  hold off;
  saveas(gcf, nombre_archivo);
end

function contar_pixeles = contar_pixeles_blancos(imagen)
   contar_pixeles = sum(imagen(:));
end

function cant_galletitas = contar_galletitas(img_todas, img_sola)
  cant_sola = contar_pixeles_blancos(img_sola);
  cant_todas = contar_pixeles_blancos(img_todas);
  cant_galletitas = round(cant_todas/cant_sola);
end

function mascara = crear_mascara(colores_hsv, tolerancia, imagen)
  mascara =  (imagen(:,:,1) >= colores_hsv(1) - tolerancia(1) & imagen(:,:,1) <= colores_hsv(1) + tolerancia(1)) & ...
             (imagen(:,:,2) >= colores_hsv(2) - tolerancia(2) & imagen(:,:,2) <= colores_hsv(2) + tolerancia(2)) & ...
             (imagen(:,:,3) >= colores_hsv(3) - tolerancia(3) & imagen(:,:,3) <= colores_hsv(3) + tolerancia(3));
end

function mascara_bn = crear_mascara_bn(imagen)
  mascara_bn = uint8(imagen) * 255;
end

function cant_sabor = contar_sabor(img_total, img_simple, color, tolerancia)
  img_hsv_total = rgb2hsv(img_total);
  img_hsv_simple = rgb2hsv(img_simple);
  mascara_total = crear_mascara(color, tolerancia, img_hsv_total);
  mascara_simple = crear_mascara(color, tolerancia, img_hsv_simple);
  mascara_bn_total = crear_mascara_bn(mascara_total);
  mascara_bn_simple = crear_mascara_bn(mascara_simple);

  cant_sabor = contar_galletitas(mascara_bn_total, mascara_bn_simple);

end

% Cargo las imagenes de solo galletita
##--------------------BAGLEY-------------------------------------
img_rellena_bagley = imread('imagenes/Simples/redonda-rellena-bagley.jpg');
img_sonrisa_bagley = imread('imagenes/Simples/redonda-sonrisa-bagley.jpg');
img_simple_bagley = imread('imagenes/Simples/redonda-simple-bagley.jpg');
img_doble_bagley = imread('imagenes/Simples/redonda-doble-bagley.jpg');
img_chips_bagley = imread('imagenes/Simples/redonda-chips-bagley2.jpg');
img_cuadrada_simple_bagley = imread('imagenes/Simples/cuadrada-simple-bagley.jpg');
img_cuadrada_doble_bagley = imread('imagenes/Simples/cuadrada-dobles-bagley.jpg');

##--------------------ARCOR-------------------------------------
img_vainilla_doble_arcor = imread('imagenes/Simples/redonda-vainilla-doble-arcor.jpg');
img_simple_arcor = imread('imagenes/Simples/redonda-simple-arcor.jpg');
img_doble_arcor = imread('imagenes/Simples/redonda-doble-arcor.jpg');
img_chips_arcor = imread('imagenes/Simples/redonda-chips-arcor.jpg');
img_anillo_arcor = imread('imagenes/Simples/anillo-chocolate-arcor.jpg');

% Cargo las imagenes
##--------------------BAGLEY-------------------------------------
img_rellenas_bagley  = imread('imagenes/Todas/redondas-rellenas-bagley.jpg');
img_sonrisas_bagley = imread('imagenes/Todas/redondas-sonrisas-bagley.jpg');
img_simples_bagley = imread('imagenes/Todas/redondas-simples-bagley.jpg');
img_dobles_bagley = imread('imagenes/Todas/redondas-dobles-bagley.jpg');
img_chipss_bagley = imread('imagenes/Todas/redondas-chips-bagley.jpg');
img_cuadradas_simple_bagley = imread('imagenes/Todas/cuadradas-simples-bagley.jpg');
img_cuadradas_doble_bagley = imread('imagenes/Todas/cuadradas-dobles-bagley.jpg');

##--------------------ARCOR-------------------------------------
img_vainilla_dobles_arcor = imread('imagenes/Todas/redondas-vainilla-dobles-arcor.jpg');
img_simples_arcor = imread('imagenes/Todas/redondas-simples-arcor.jpg');
img_dobles_arcor = imread('imagenes/Todas/redondas-dobles-arcor.jpg');
img_chipss_arcor = imread('imagenes/Todas/redondas-chips-arcor.jpg');
img_anillos_arcor = imread('imagenes/Todas/anillos-arcor.jpg');

##----------------------Colores------------------------------
color_marron = [0.488, 0.264, 0.208];
tolerancia_marron = [0.55, 0.4, 0.35];

color_amarillo = [0.1111, 0.507, 0.573];
color_blanco = [0.91, 0.09, 0.8];
color_rosa = [0.8472, 0.15, 0.75];

tolerancia_rosa = [0.3, 0.3, 0.3];
tolerancia_blanca = [0.3, 0.2, 0.2];
tolerancia_amarillo = [0.3, 0.5, 0.3];

% Aplico el filtro a las imagenes
##--------------------BAGLEY-------------------------------------
img_filtered_rellena_bagley = apply_all_filter(img_rellena_bagley, 5, 5, 7, 0.1, 0.25, 5);
img_filtered_rellenas_bagley = apply_all_filter(img_rellenas_bagley, 5, 5, 7, 0.1, 0.25, 5);

img_filtered_sonrisa_bagley = apply_all_filter(img_sonrisa_bagley, 7, 5, 7, 0.1, 0.15, 7);
img_filtered_sonrisas_bagley = apply_all_filter(img_sonrisas_bagley, 7, 5, 7, 0.1, 0.15, 7);

img_filtered_simple_bagley = apply_all_filter(img_simple_bagley, 7, 5, 7, 0.1, 0.15, 7);
img_filtered_simples_bagley = apply_all_filter(img_simples_bagley, 7, 5, 7, 0.1, 0.15, 7);

img_filtered_doble_bagley = apply_all_filter(img_doble_bagley, 3, 2, 3, 0.1, 0.3, 3);
img_filtered_dobles_bagley = apply_all_filter(img_dobles_bagley, 7, 5, 7, 0.1, 0.15, 3);

img_filtered_chips_bagley = apply_all_filter(img_chips_bagley, 3, 2, 3, 0.1, 0.15, 3);
img_filtered_chipss_bagley = apply_all_filter(img_chipss_bagley, 3, 2, 3, 0.1, 0.15, 3);

img_filtered_cuadrada_simple = apply_all_filter(img_cuadrada_simple_bagley, 3, 3, 3, 0.1, 0.15, 3);
img_filtered_cuadradas_simple = apply_all_filter(img_cuadradas_simple_bagley, 3, 3, 3, 0.1, 0.15, 3);

img_filtered_cuadrada_doble = apply_all_filter(img_cuadrada_doble_bagley, 3, 3, 3, 0.1, 0.15, 3);
img_filtered_cuadradas_doble = apply_all_filter(img_cuadradas_doble_bagley, 3, 3, 3, 0.1, 0.15, 3);

####--------------------ARCOR-------------------------------------
img_filtered_vainilla_doble_arcor = apply_all_filter(img_vainilla_doble_arcor, 5, 5, 7, 0.1, 0.15, 5);
img_filtered_vainilla_dobles_arcor = apply_all_filter(img_vainilla_dobles_arcor, 5, 5, 7, 0.1, 0.13, 5);

img_filtered_simple_arcor = apply_all_filter(img_simple_arcor, 7, 5, 7, 0.1, 0.15, 7);
img_filtered_simples_arcor = apply_all_filter(img_simples_arcor, 7, 5, 7, 0.1, 0.15, 7);

img_filtered_doble_arcor = apply_all_filter(img_doble_arcor, 5, 5, 7, 0.1, 0.25, 5);
img_filtered_dobles_arcor = apply_all_filter(img_dobles_arcor, 5, 5, 7, 0.1, 0.15, 5);

img_filtered_chips_arcor = apply_all_filter(img_chips_arcor, 7, 1, 7, 0.1, 0.15, 7);
img_filtered_chipss_arcor = apply_all_filter(img_chipss_arcor, 7, 1, 7, 0.1, 0.15, 7);

img_filtered_anillo_arcor = apply_all_filter(img_anillo_arcor, 5, 5, 5, 0.1, 0.15, 5);
img_filtered_anillos_arcor = apply_all_filter(img_anillos_arcor, 5, 5, 5, 0.1, 0.15, 5);

####--------------------BAGLEY-------------------------------------
##fprintf("Método contando las elipsis:\n");
##fprintf("Galletitas bagley:\n");
##cant_elipsis_rellenas_bagley = draw_elipses(img_filtered_rellena_bagley,img_filtered_rellenas_bagley, img_rellenas_bagley, 'imagenes/Procesadas/rellenas_bagley.png');
##fprintf("Cantidad de rellenas son %d\n", cant_elipsis_rellenas_bagley);
##
##cant_elipsis_sonrisas = draw_elipses(img_filtered_sonrisa_bagley, img_filtered_sonrisas_bagley, img_sonrisas_bagley, 'imagenes/Procesadas/sonrisas.png');
##fprintf("Cantidad de sonrisas son %d\n", cant_elipsis_sonrisas);
##
##cant_elipsis_simple_bagley = draw_elipses(img_filtered_simple_bagley, img_filtered_simples_bagley, img_simples_bagley, 'imagenes/Procesadas/simple_bagley.png');
##fprintf("Cantidad de simples son %d\n", cant_elipsis_simple_bagley);
##
##cant_elipsis_dobles_bagley = draw_elipses(img_filtered_doble_bagley, img_filtered_dobles_bagley, img_dobles_bagley, 'imagenes/Procesadas/dobles_bagley.png'); %ver
##fprintf("Cantidad de dobles son %d\n", cant_elipsis_dobles_bagley);
##
##cant_elipsis_chips = draw_elipses(img_filtered_chips_bagley, img_filtered_chipss_bagley, img_chipss_bagley, 'imagenes/Procesadas/chips_bagley.png'); %ver
##fprintf("Cantidad de chips son %d\n", cant_elipsis_chips);
##
##draw_lines(img_filtered_cuadrada_simple, img_cuadrada_simple_bagley, 'imagenes/Procesadas/cuadrada_simple.png');
##draw_lines(img_filtered_cuadradas_simple, img_cuadradas_simple_bagley, 'imagenes/Procesadas/cuadradas_simple.png');
##
##draw_lines(img_filtered_cuadrada_doble, img_cuadrada_doble_bagley, 'imagenes/Procesadas/cuadrada_doble.png');
##draw_lines(img_filtered_cuadradas_doble, img_cuadradas_doble_bagley, 'imagenes/Procesadas/cuadradas_doble.png');
##
####--------------------ARCOR-------------------------------------
##fprintf("\nGalletitas arcor:\n");
##cant_elipsis_doble_arcor = draw_elipses(img_filtered_vainilla_doble_arcor, img_filtered_vainilla_dobles_arcor, img_vainilla_dobles_arcor, 'imagenes/Procesadas/doble_arcor.png');
##fprintf("Cantidad de dobles son %d\n", cant_elipsis_doble_arcor);
##
##cant_elipsis_simple_arcor = draw_elipses(img_filtered_simple_arcor, img_filtered_simples_arcor, img_simples_arcor, 'imagenes/Procesadas/simple_arcor.png'); %VERRRRRRRR
##fprintf("Cantidad de simples son %d\n", cant_elipsis_simple_arcor);
##
##cant_elipsis_doble_arcor = draw_elipses(img_filtered_doble_arcor, img_filtered_dobles_arcor, img_dobles_arcor, 'imagenes/Procesadas/doble_arcor.png');
##fprintf("Cantidad de doble son %d\n", cant_elipsis_doble_arcor);
##
##cant_elipsis_chips_arcor = draw_elipses(img_filtered_chips_arcor, img_filtered_chipss_arcor, img_chipss_arcor, 'imagenes/Procesadas/chips_arcor.png'); %VERRRRRRRRR
##fprintf("Cantidad de anillos son %d\n", cant_elipsis_chips_arcor);
##
##cant_elipsis_anillos = draw_elipses(img_filtered_anillo_arcor, img_filtered_anillos_arcor, img_anillos_arcor, 'imagenes/Procesadas/anillos.png');
##fprintf("Cantidad de anillos son %d\n", cant_elipsis_anillos);

##--------------------BAGLEY-------------------------------------
fprintf("\nContado los pixeles:\n");
fprintf("Galletitas bagley:\n");
cant_rellenas_bagley = contar_galletitas(img_filtered_rellenas_bagley, img_filtered_rellena_bagley);
fprintf("Cantidad de galletitas rellenas son %d\n", cant_rellenas_bagley);

cant_sonrisas_bagley = contar_galletitas(img_filtered_sonrisas_bagley, img_filtered_sonrisa_bagley);
fprintf("Cantidad de galletitas sonrisas son %d\n", cant_sonrisas_bagley);

cant_simples_bagley = contar_galletitas(img_filtered_simples_bagley, img_filtered_simple_bagley);
fprintf("Cantidad de galletitas simples son %d\n", cant_simples_bagley);

cant_chips_bagley = contar_galletitas(img_filtered_chipss_bagley, img_filtered_chips_bagley);
fprintf("Cantidad de galletitas con chips son %d\n", cant_chips_bagley);

%USAR OTRO FILTRO
cant_dobles_bagley = contar_galletitas(img_filtered_dobles_bagley, img_filtered_doble_bagley);
fprintf("Cantidad de galletitas dobles son %d\n", cant_dobles_bagley); %ver

cant_cuadradas_simple = contar_galletitas(img_filtered_cuadradas_simple, img_filtered_cuadrada_simple);
fprintf("Cantidad de galletitas cuadradas simple son %d\n", cant_cuadradas_simple);

cant_cuadradas_dobles = contar_galletitas(img_filtered_cuadradas_doble, img_filtered_cuadrada_doble);
fprintf("Cantidad de galletitas cuadradas dobles son %d\n", cant_cuadradas_dobles);

##--------------------ARCOR-------------------------------------
fprintf("\nGalletitas arcor:\n");
cant_vainillas_arcor = contar_galletitas(img_filtered_vainilla_dobles_arcor, img_filtered_vainilla_doble_arcor);
fprintf("Cantidad de galletitas de vainilla son %d\n", cant_vainillas_arcor);

cant_simples_arcor = contar_galletitas(img_filtered_simples_arcor, img_filtered_simple_arcor);
fprintf("Cantidad de galletitas simples son %d\n", cant_simples_arcor);

cant_dobles_arcor = contar_galletitas(img_filtered_dobles_arcor, img_filtered_doble_arcor);
fprintf("Cantidad de galletitas dobles son %d\n", cant_dobles_arcor);

cant_chips_arcor = contar_galletitas(img_filtered_chipss_arcor, img_filtered_chips_arcor);
fprintf("Cantidad de galletitas con chips son %d\n", cant_chips_arcor);

cant_anillos_arcor = contar_galletitas(img_filtered_anillos_arcor, img_filtered_anillo_arcor);
fprintf("Cantidad de anillos son %d\n", cant_anillos_arcor);

##--------------------BAGLEY-------------------------------------
cant_vainillas_bagley = 0;
cant_chocolate_bagley = 0;
cant_frutilla_bagley = 0;
cant_blancas_bagley = 0;

cant_vainillas_bagley = contar_sabor(img_rellenas_bagley, img_rellena_bagley, color_amarillo, tolerancia_amarillo) + cant_vainillas_bagley;
cant_vainillas_bagley = contar_sabor(img_sonrisas_bagley, img_sonrisa_bagley, color_amarillo, tolerancia_amarillo) + cant_vainillas_bagley;
cant_vainillas_bagley = contar_sabor(img_chipss_bagley, img_chips_bagley, color_amarillo, tolerancia_amarillo) + cant_vainillas_bagley;
cant_vainillas_bagley = contar_sabor(img_simples_bagley, img_simple_bagley, color_amarillo, tolerancia_amarillo) + cant_vainillas_bagley;
fprintf("Cantidad de galletitas de vainilla de Bagley son %d\n", cant_vainillas_bagley);

cant_chocolate_bagley = contar_sabor(img_cuadradas_simple_bagley, img_cuadrada_simple_bagley, color_marron, tolerancia_marron) + cant_chocolate_bagley;
cant_chocolate_bagley = contar_sabor(img_cuadradas_doble_bagley, img_cuadrada_doble_bagley, color_marron, tolerancia_marron) + cant_chocolate_bagley;
cant_chocolate_bagley = contar_sabor(img_dobles_bagley, img_doble_bagley, color_marron, tolerancia_marron) + cant_chocolate_bagley;
fprintf("Cantidad de galletitas de Chocolate de Bagley son %d\n", cant_chocolate_bagley);

cant_frutilla_bagley = contar_sabor(img_simples_bagley, img_simple_bagley, color_rosa, tolerancia_rosa) + cant_frutilla_bagley;
fprintf("Cantidad de galletitas de Frutilla de Bagley son %d\n", cant_frutilla_bagley);

fprintf("Cantidad de galletitas de Blancas de Bagley son %d\n", cant_blancas_bagley);

####--------------------ARCOR-------------------------------------
canti_vainillas_arcor = 0;
cant_chocolate_arcor = 0;
cant_frutilla_arcor = 0;
cant_blancas_arcor = 0;

canti_vainillas_arcor = contar_sabor(img_anillos_arcor, img_anillo_arcor, color_amarillo, tolerancia_amarillo) + canti_vainillas_arcor;
canti_vainillas_arcor = contar_sabor(img_chipss_arcor, img_chips_arcor, color_amarillo, tolerancia_amarillo) + canti_vainillas_arcor;
canti_vainillas_arcor = contar_sabor(img_simples_arcor, img_simple_arcor, color_amarillo, tolerancia_amarillo) + canti_vainillas_arcor;
fprintf("Cantidad de galletitas de Vainilla de Arcor son %d\n", canti_vainillas_arcor);

cant_chocolate_arcor = contar_sabor(img_dobles_arcor, img_doble_arcor, color_marron, tolerancia_marron) + cant_chocolate_arcor;
fprintf("Cantidad de galletitas de Chocolate de Arcor son %d\n", cant_chocolate_arcor);

cant_frutilla_arcor = contar_sabor(img_anillos_arcor, img_anillo_arcor, color_rosa, tolerancia_rosa) + cant_frutilla_arcor;
fprintf("Cantidad de galletitas de Frutilla de Arcor son %d\n", cant_frutilla_arcor);

cant_blancas_arcor = contar_sabor(img_anillos_arcor, img_anillo_arcor, color_blanco, tolerancia_blanca) + cant_blancas_arcor;
fprintf("Cantidad de galletitas de Blancas de Arcor son %d\n", cant_blancas_arcor);
