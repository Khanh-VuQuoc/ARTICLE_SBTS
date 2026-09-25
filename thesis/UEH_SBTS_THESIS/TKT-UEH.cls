\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{TKT-UEH}[2025/10/29 v2.1 UEH Thesis class]
\ProcessOptions\relax
\LoadClass[oneside]{book}
\RequirePackage[fontsize=13pt]{scrextend}
\RequirePackage{geometry}
\geometry{a4paper, left=35mm, right=20mm, top=35mm, bottom=30mm}
\RequirePackage[utf8]{inputenc}
\RequirePackage[vietnamese]{babel}
\RequirePackage{fontspec}
\RequirePackage{polyglossia}
\setmainlanguage{vietnamese}
\RequirePackage{titlesec}
\RequirePackage[backend=biber,style=apa,sortcites=true,uniquename=false,natbib=true]{biblatex}
\setmainfont{Times New Roman}[
  Scale=1.09,
  WordSpace={1,1,1},
  Ligatures=TeX
]
\RequirePackage{lmodern,sectsty,tabularx,fancyhdr,array,booktabs,threeparttable,xcolor,float}
\RequirePackage{amsmath,amssymb,caption,tikz,etoolbox,titlesec,enumitem,hyperref,tocloft,indentfirst}
\hypersetup{colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue}
\linespread{1.5}
\pagestyle{fancy}\fancyhf{} \fancyhead[L]{\nouppercase{\leftmark}} \fancyhead[R]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}\renewcommand{\footrulewidth}{0pt}
\captionsetup{labelfont=bf}

\makeatletter
\renewcommand\@makechapterhead[1]{%
  \vspace*{-6pt}%
  {\par\centering
   \normalfont\bfseries\Large
   CHƯƠNG~\thechapter.\par
   \vspace{0.5em}%
   \MakeUppercase{#1}\par
   \vspace{24pt}%
  }%
  \normalfont\normalsize\normalcolor
  \par
}

\let\TKT@orig@l@chapter\l@chapter
\renewcommand*\l@chapter[2]{%
  \begingroup
    \def\numberline##1{CHƯƠNG~##1.\enspace}%
    \TKT@orig@l@chapter{#1}{#2}%
  \endgroup
}
\makeatother

\addto\captionsvietnamese{\renewcommand{\contentsname}{\begin{center}\Large MỤC LỤC \end{center}}\renewcommand{\listfigurename}{\begin{center}\Large DANH MỤC HÌNH \end{center}}\renewcommand{\listtablename}{\begin{center}\Large DANH MỤC BẢNG \end{center}}\renewcommand{\bibname}{\begin{center}\Large TÀI LIỆU THAM KHẢO\end{center}}}
\newcommand{\TKTStudentName}{TÊN HỌC VIÊN}
\newcommand{\TKTThesisTitle}{TÊN ĐỀ TÀI}
\newcommand{\TKTSpec}{Toán kinh tế}
\newcommand{\TKTAdvisor}{TÊN GIẢNG VIÊN HƯỚNG DẪN}
\newcommand{\TKTCity}{Thành phố Hồ Chí Minh}
\newcommand{\TKTYear}{\the\year}
\newcommand{\TKTMon}{\the\month}
\newcommand{\TKTDay}{\the\day}
\newcommand{\setTKTStudentName}[1]{\renewcommand{\TKTStudentName}{#1}}
\newcommand{\setTKTThesisTitle}[1]{\renewcommand{\TKTThesisTitle}{#1}}
\newcommand{\setTKTThesisType}[1]{\renewcommand{\TKTThesisType}{#1}}
\newcommand{\setTKTSpec}[1]{\renewcommand{\TKTSpec}{#1}}
\newcommand{\setTKTStudentID}[1]{\renewcommand{\TKTStudentID}{#1}}
\newcommand{\setTKTAdvisor}[1]{\renewcommand{\TKTAdvisor}{#1}}
\newcommand{\setTKTCity}[1]{\renewcommand{\TKTCity}{#1}}
\newcommand{\setTKTYear}[1]{\renewcommand{\TKTYear}{#1}}
\newcommand{\setTKTDay}[1]{\renewcommand{\TKTDay}{#1}}
\newcommand{\setTKTMon}[1]{\renewcommand{\TKTMon}{#1}}
\newcommand{\TKTDrawFrame}{\begin{tikzpicture}[remember picture, overlay]\draw[line width=0.8pt]($(current page.north west)+(17mm,-17mm)$)rectangle($(current page.south east)+(-17mm,17mm)$);\end{tikzpicture}}
\newcommand{\TKTCover}{\thispagestyle{empty}\TKTDrawFrame\vspace{-2cm}\begin{center}{\large \textbf{BỘ GIÁO DỤC VÀ ĐÀO TẠO}}\\[6pt]{\large \textbf{ĐẠI HỌC KINH TẾ THÀNH PHỐ HỒ CHÍ MINH}}\\[70pt]{\large \textbf{\MakeUppercase{\TKTStudentName}}}\\[65pt]{\LARGE \textbf{\MakeUppercase{\TKTThesisTitle}}}\\[60pt]{\large \textbf{ĐỀ ÁN TỐT NGHIỆP THẠC SĨ}}\vfill{\normalsize \TKTCity{} -- Năm \TKTYear}\end{center}\clearpage}
\newcommand{\TKTTitle}{\thispagestyle{empty}\TKTDrawFrame\vspace{-2.5cm}\begin{center}{\large \textbf{BỘ GIÁO DỤC VÀ ĐÀO TẠO}}\\[6pt]{\large \textbf{ĐẠI HỌC KINH TẾ THÀNH PHỐ HỒ CHÍ MINH}}\\[50pt]{\large \textbf{\MakeUppercase{\TKTStudentName}}}\\[45pt]{\LARGE \textbf{\MakeUppercase{\TKTThesisTitle}}}\end{center}\vspace{4mm}\noindent\begin{tabular}{@{}l l l@{}}\hspace{3cm} & Chuyên ngành: & Toán Kinh tế \\& Hướng đào tạo: & Hướng ứng dụng \\& Mã số: &830108  \end{tabular} \\[35pt]\begin{center}{\large \textbf{ĐỀ ÁN TỐT NGHIỆP THẠC SĨ}} \\ [50pt] NGƯỜI HƯỚNG DÂN KHOA HỌC \\[10pt]{\large \textbf{\MakeUppercase{\TKTAdvisor}}} \vfill{\normalsize Thành phố Hồ Chí Minh -- Năm \TKTYear}\end{center}\clearpage}
\newcommand{\TKTdeclare}{Tôi cam đoan đây là công trình nghiên cứu của riêng tôi. Các số liệu, kết quả và trích dẫn trong luận văn đều được nêu rõ nguồn gốc và trung thực. Luận văn chưa từng được nộp để nhận bất kỳ học vị nào tại bất cứ cơ sở đào tạo nào.}
\newcommand{\setTKTdeclar}[1]{\gdef\TKTdeclare{#1}}
\newcommand{\TKTmakedeclare}{\thispagestyle{empty}\begin{center}{\Large \textbf{LỜI CAM ĐOAN}}\\[24pt]\end{center}\noindent \TKTdeclare \begin{flushright}\textit{\TKTCity, ngày~\TKTDay~tháng~\TKTMon~năm~\TKTYear}\\[8pt]\textbf{Tác giả}\\[40pt]\textbf{\TKTStudentName}\end{flushright}\clearpage}
\newenvironment{TKTAbbrev}{%
  \clearpage
  \thispagestyle{empty}%
  \begin{center}
    {\Large \textbf{DANH MỤC CÁC TỪ VIẾT TẮT}}%
  \end{center}%
 \vspace{12pt}%
  \begin{tabular}{@{}p{0.25\textwidth}p{0.7\textwidth}@{}}%
}{%
  \end{tabular}%
  \clearpage
}

\newenvironment{TKTAbstractVI}{\clearpage\thispagestyle{empty}\begin{center}\Large\textbf{TÓM TẮT}\end{center}\vspace{12pt}}{\par\clearpage}
\newenvironment{TKTAbstractEN}{\clearpage\thispagestyle{empty}\begin{center}\Large\textbf{ABSTRACT}\end{center}\vspace{12pt}}{\par\clearpage}

\makeatletter
\renewcommand{\l@figure}[2]{%
  \@dottedtocline{1}{1.5em}{3.5em}{\figurename.\hspace{0.1em}#1}{#2}%
}
\renewcommand{\l@table}[2]{%
  \@dottedtocline{1}{1.5em}{3.5em}{\tablename.\hspace{0.1em}#1}{#2}%
}
\makeatother


\newcommand{\TKTInput}[1]{\IfFileExists{#1}{\input{#1}}{}}
\makeatletter
\newcommand{\TKTtrangdau}{%
  \pagenumbering{gobble}
  \TKTCover
  \TKTTitle
  \TKTmakedeclare
\begingroup
 \par\vspace*{-2.5cm}
\tableofcontents
\thispagestyle{empty}
\renewcommand{\cftaftertoctitle}{\vspace{1em}} % bỏ gạch dưới
\renewcommand{\cftbeforetoctitleskip}{0pt}
 \markboth{}{}% xoá marks để trang sau không hiện "MỤC LỤC"
    \@starttoc{toc}
\endgroup
\clearpage
  \IfFileExists{trangbia/abbrev.tex}{%
    \input{trangbia/abbrev.tex}%
    \clearpage
  }{}%
  \begingroup
    \par\vspace*{-2.5cm}   % chỉnh -1cm, -1.2cm, -2cm... cho vừa mắt
    \listoftables
  \endgroup
  \clearpage
  \begingroup
    \par\vspace*{-2.5cm}   % chỉnh -1cm, -1.2cm, -2cm... cho vừa mắt
    \listoffigures
  \endgroup
  \clearpage
 \mainmatter
  \setcounter{chapter}{0}% đảm bảo chương đầu = 1
  \pagenumbering{arabic}
\setcounter{page}{1}
}

\makeatother
\cleardoublepage
\addcontentsline{toc}{chapter}{TRANG PHỤ BÌA}
\addcontentsline{toc}{chapter}{LỜI CAM ĐOAN}
\addcontentsline{toc}{chapter}{DANH MỤC CÁC TỪ VIẾT TẮT}
\addcontentsline{toc}{chapter}{DANH MỤC CÁC BẢNG}
\addcontentsline{toc}{chapter}{DANH MỤC CÁC HÌNH}
\addcontentsline{toc}{chapter}{TÓM TẮT – ABSTRACT}


\DeclareLanguageMapping{english}{english-apa}
\DeclareLanguageMapping{vietnamese}{english-apa}
\DefineBibliographyStrings{english}{references={TÀI LIỆU THAM KHẢO},  andothers  = {và cộng sự}, and =  {và}}
\DefineBibliographyStrings{vietnamese}{references={TÀI LIỆU THAM KHẢO},andothers  = {và cộng sự}, and =  {và}}
\AtBeginBibliography{\pagestyle{empty}}
\newcommand{\printrefs}{%
  \clearpage
  \phantomsection
  \chapter*{\begin{center} \Large TÀI LIỆU THAM KHẢO \end{center}}%
  \thispagestyle{empty}% <-- bỏ số trang ở chính trang tiêu đề
  \addtocontents{toc}{%
    \protect\contentsline{chapter}{TÀI LIỆU THAM KHẢO}{}{}%
  }%
  \begingroup
    \pagestyle{empty}% <
    \printbibliography[heading=none]%
  \endgroup
  \clearpage
  \pagestyle{plain}
}

\makeatletter
\newcommand{\TKT@phuluc@toctype}{chapter}%
\@ifundefined{chapter}{\renewcommand{\TKT@phuluc@toctype}{section}}{}%

\newcounter{tktappendix}
\renewcommand{\thetktappendix}{\Alph{tktappendix}}
\newif\ifTKT@firstappendix

\newenvironment{TKTPhuluc}{%
  \clearpage
  \begingroup
    \let\ps@plain\ps@empty
    \@ifundefined{ps@fancy}{}{\let\ps@fancy\ps@empty}
    \let\ps@headings\ps@empty
    \let\ps@myheadings\ps@empty
    \pagestyle{empty}%
    \@ifundefined{headrulewidth}{}{\renewcommand{\headrulewidth}{0pt}}%
    \@ifundefined{plainheadrulewidth}{}{\renewcommand{\plainheadrulewidth}{0pt}}%
    \@ifundefined{footrulewidth}{}{\renewcommand{\footrulewidth}{0pt}}%
    \markboth{}{}%
    \let\thispagestyle\@gobble
    \pagenumbering{gobble}%
    \phantomsection
    \addtocontents{toc}{%
    \protect\contentsline{chapter}{PHỤ LỤC}{}{}%
  }%
    \par{\centering{\bfseries\Large PHỤ LỤC}\par}\normalfont\vspace{0.5em}%
    \TKT@firstappendixtrue
}{%
  \endgroup
  \clearpage
}

\newcommand{\TKT@noopsection}[1]{\par\noindent{\bfseries #1}\par\normalfont\medskip}

\newcommand{\TKTfilePL}[2]{%
  \ifTKT@firstappendix \TKT@firstappendixfalse \else \par\bigskip \fi
  \stepcounter{tktappendix}%
  \par\noindent{\centering{\bfseries\large PHỤ LỤC~\thetktappendix\ --\ #1}\par}\normalfont\vspace{0.5em}%
  {\begingroup
    \setcounter{secnumdepth}{-1}%
    \renewcommand\thesection{}%
    \renewcommand\thesubsection{}%
    \renewcommand\thesubsubsection{}%
    \let\chapter\TKT@noopsection
    \expandafter\let\csname chapter*\endcsname\TKT@noopsection
    \let\section\TKT@noopsection
    \expandafter\let\csname section*\endcsname\TKT@noopsection
    \let\subsection\TKT@noopsection
    \expandafter\let\csname subsection*\endcsname\TKT@noopsection
    \let\subsubsection\TKT@noopsection
    \expandafter\let\csname subsubsection*\endcsname\TKT@noopsection
    \input{#2}%
  \endgroup}%
}
\makeatother
\endinput
