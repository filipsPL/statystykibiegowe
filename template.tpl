<!DOCTYPE html>
<html lang="pl">
	<head>
		<meta charset="utf-8">
		<meta http-equiv="X-UA-Compatible" content="IE=edge">
		<title>Statystyki biegowe: $TYTUL</title>
		<meta name="viewport" content="width=device-width, initial-scale=1">
		<meta name="Description" lang="pl" content="Statystyki biegowe">
		<meta name="author" content="Filip Stefaniak">
		<meta name="robots" content="index, follow">
		<meta http-equiv="content-type" content="text/html; charset=UTF-8" />

		<!-- Override CSS file - add your own CSS rules -->
		<link rel="stylesheet" href="${RELATIVE}styles.css">
	</head>
	<body>
		<div class="header">
			<div class="container">
				<h1 class="header-heading"><img src="${RELATIVE}runner48.png" alt="runner" /> Statystyki biegowe i tri</h1>
			</div>
		</div>
		<div class="nav-bar">
			<div class="container">
				<ul class="nav">
					<li><a href="${RELATIVE}index.html">Strona główna</a></li>
					<li><a href="${RELATIVE}o.html">O stronie</a></li>
					<li><a href="${RELATIVE}kontakt.html">Kontakt</a></li>
				</ul>
			</div>
		</div>
		<div class="content">
			<div class="container">
				<div class="main">
				<h1>$TYTUL</h1>
$TRESC
				</div>

<div class="aside">
<h2>Menu</h2>
<p>$MENU</p>
<p>Dystans: $DYSTANS km</p>
<p>Liczba uczestników: $UCZESTNIKOW</p>
<p class="info1">Wygenerowano: $DATA</p>
</div>

			</div>
		</div>

		<div class="footer">
			<div class="container">
				Wymyślił, napisał, policzył i zaprogramował: filips | <a href="https://filipspl.github.io/statystykibiegowe/">https://filipspl.github.io/statystykibiegowe/</a>
			</div>
		</div>
	</body>
</html>
