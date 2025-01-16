pkg load image;

img = imread('Imagenes/fotomia.jpg');
%img = imread('Imagenes/fotomia2.jpg');
%img = imread('Imagenes/fotointernet3.jpg');
%img = imread('Imagenes/fotointernet2.jpeg');
%img = imread('Imagenes/fotointernet.webp');

gray_img = rgb2gray(img);

kernel_size = 5; % Tamaño del filtro
sigma = 2; % Desviación estándar
gaussian_filter = fspecial('gaussian', kernel_size, sigma);
img_filtered = imfilter(gray_img, gaussian_filter, 'symmetric');

% Detecto los bordes
edges = edge(img_filtered, 'canny');

% Utilizo la transformada de Hough
[H, theta, rho] = hough(edges);

% Encontrar los picos de la Transformada de Hough
peaks = houghpeaks(H, 5, 'threshold', ceil(0.3 * max(H(:))));

% Obtengo las líneas correspondientes a los picos
lines = houghlines(edges, theta, rho, peaks);

% Selecciono la línea más larga
max_len = 0;
for k = 1:length(lines)
    len = norm(lines(k).point1 - lines(k).point2);
    if len > max_len
        max_len = len;
        longest_line = lines(k);
    end
end

% Guardo las coordenadas
p1 = longest_line.point1;
p2 = longest_line.point2;

% Calculo el ángulo de inclinación de la línea respecto al eje X
angle = atan2d(p2(2) - p1(2), p2(1) - p1(1));

 % Roto la imagen respecto al ángulo
rotated_img = imrotate(img, angle, 'bilinear', 'crop');

if angle > 90 || angle < -90
  rotated_img = imrotate(rotated_img, 180, 'bilinear', 'crop');
end

% Muestro la imagen original con la linea y la imagen rotada
figure;
subplot(1, 2, 1);
imshow(img);
title('Imagen original con la línea marcada');
hold on;
line([p1(1), p2(1)], [p1(2), p2(2)], 'Color', 'red', 'LineWidth', 2);
hold off;

subplot(1, 2, 2);
imshow(rotated_img);
title('Imagen rotada respecto a la línea');
