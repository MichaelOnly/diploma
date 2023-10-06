\ProvidesClass{disser}[2014/10/11 Dissertation Class by Ilya Stupakov]
\LoadClass[a4paper,
	fontsize=14pt,
	headings=small,
	listof=leveldown,
	chapterentrydots=true,
	numbers=endperiod,
	%chapterprefix=true,
	%draft,
]{scrreprt}



\RequirePackage{iftex}
\ifxetex
	\RequirePackage{fontspec}                         % подготавливает загрузку шрифтов Open Type, True Type и др.
	\defaultfontfeatures{Ligatures={TeX}}             % свойства шрифтов по умолчанию
	\setmainfont{Times New Roman Cyr}                     % задаёт основной шрифт документа
	\setsansfont{Comic Sans MS}                       % задаёт шрифт без засечек
	\setmonofont{Courier New}                         % задаёт моноширинный шрифт
\else
	\RequirePackage{cmap}                             % Улучшенный поиск русских слов в полученном pdf-файле
	\RequirePackage[T2A]{fontenc}                     % Поддержка русских букв
	\RequirePackage[utf8]{inputenc}                   % Кодировка utf8
\fi

\RequirePackage[english,russian]{babel}                   % загружает пакет многоязыковой вёрстки

\RequirePackage[hidelinks, unicode=true, bookmarksnumbered=true]{hyperref}
\makeatletter
\renewcommand{\Hy@numberline}[1]{#1.} % Добавляем точки в оглавлении PDF
\makeatother

\RequirePackage{amsmath,amsfonts,amssymb,amsthm,mathtools} % AMS
\RequirePackage[usenames,dvipsnames,svgnames,table,rgb]{xcolor}

\RequirePackage{indentfirst} % Красная строка
\setlength{\parindent}{1cm} % Отступ для абзаца
\RequirePackage{setspace}

\RequirePackage{tabularx}
\RequirePackage{pdfpages}

%%% Страница
\RequirePackage[top=20mm,bottom=20mm,left=30mm,right=10mm,footskip=1cm,headsep=0.7cm]{geometry} % Поля страницы

%%% Выравнивание и переносы %%%

\sloppy                            % Избавляемся от переполнений
\clubpenalty=10000                 % Запрещаем разрыв страницы после первой строки абзаца
\widowpenalty=10000                % Запрещаем разрыв страницы после последней строки абзаца

%%% Настройка шрифтов и заголовков %%%

\renewcommand{\chapterformat}{\chapapp~\thechapter\autodot\enskip}% Добавляем префиксы для глав и приложений
%font={\let\@hangfrom\relax}
\RedeclareSectionCommand[beforeskip=0cm,tocnumwidth=1cm,font={\bfseries\Large}]{chapter}
\RedeclareSectionCommands[font={\bfseries}]{section,subsection}
\let\raggedchapter\centering%Center all chapterheads
\let\raggedsection\centering%Center all sectionheads
\setkomafont{disposition}{\bfseries}
\setkomafont{descriptionlabel}{\bfseries}

% нумерация источников в самом списке — через точку
\makeatletter
\renewcommand\@biblabel[1]{#1. }
\makeatother

\renewcommand\addchaptertocentry[2]{% Добавляем префиксы для глав в оглавлении
  \Ifstr{#1}{}{
    \addtocentrydefault{chapter}{}{#2}
  }{
    \addtocentrydefault{chapter}{\chapapp~#1}{#2}
  }
}

\newcommand*{\appendixmore}{% Добавляем префиксы для приложений в оглавлении
  \addtocontents{toc}{
    \RedeclareSectionCommand[tocnumwidth=5.0cm]{chapter}
  }
}

%%% Библиография %%%

%\usepackage{csquotes}
% \usepackage[backend=biber,
% 	style=nature,         % стиль цитирования и библиографии             
% ]{biblatex}

%%% Переопределение именований %%%
\addto\captionsrussian{\renewcommand{\abstractname}{АННОТАЦИЯ}}
\addto\captionsrussian{\renewcommand{\appendixname}{ПРИЛОЖЕНИЕ}}
\addto\captionsrussian{\renewcommand{\bibname}{СПИСОК ЛИТЕРАТУРЫ}}
\addto\captionsrussian{\renewcommand{\chaptername}{}}
\addto\captionsrussian{\renewcommand{\contentsname}{СОДЕРЖАНИЕ}}
\addto\captionsrussian{\renewcommand{\figurename}{Рисунок}}
\addto\captionsrussian{\renewcommand{\indexname}{Предметный указатель}}
\addto\captionsrussian{\renewcommand{\listfigurename}{Список рисунков}}
\addto\captionsrussian{\renewcommand{\listtablename}{Список таблиц}}
\addto\captionsrussian{\renewcommand{\tablename}{Таблица}}

%%% Fix for some bug in biblatex %%%
%\NewBibliographyString{langjapanese} 
%\NewBibliographyString{fromjapanese}


\endinput