# ProstateCancerProgresssion
# Author : Abed Alkhateeb
++++++++++++++++++++++++
Implementation for Prostate Cancer Progression Machine Learning Model for NGS Data

Requirements:
-------------
1- Download Zseq Preprocessing Tool: https://sourceforge.net/projects/zseq/

2- Download Tuxedo Suite:
    Tophat: http://ccb.jhu.edu/software/tophat/index.shtml
    Cufflinks-CuffDiff: http://cole-trapnell-lab.github.io/cufflinks/   
    or you may run the tools using galxy project on a hosting server such as:
    usegalaxy.org

3- Download Java SE Development kit:  https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html 

4- Download Weka, Developer version: https://www.cs.waikato.ac.nz/ml/weka/

5- Using package manager in Weka, Download RerankingSearch package

Data:
-----
The Data of this roject can be found on:
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE54460
The samples are zipped in SRA Format, you need to use SRA-Toolkit to unzip them


Samples Files:
--------------
GSM1323647	Prostate_MCC-PT081
GSM1323648	Prostate_MCC-PT127
GSM1323649	Prostate_MCC-PT168
GSM1323650	Prostate_MCC-PT184
GSM1323651	Prostate_MCC-PT199
GSM1323652	Prostate_MCC-PT236
GSM1323653	Prostate_MCC-PT243
GSM1323654	Prostate_UTPC008
GSM1323655	Prostate_UTPC020
GSM1323656	Prostate_UTPC107
GSM1323657	Prostate_UTPC162
GSM1323658	Prostate_VA-PC-00-87
GSM1323659	Prostate_VA-PC-00-90
GSM1323660	Prostate_VA-PC-00-91
GSM1323661	Prostate_VA-PC-00-92
GSM1323662	Prostate_VA-PC-00-93
GSM1323663	Prostate_VA-PC-00-94
GSM1323664	Prostate_VA-PC-00-95
GSM1323665	Prostate_VA-PC-00-97
GSM1323666	Prostate_VA-PC-00-98
GSM1323667	Prostate_VA-PC-91-61
GSM1323668	Prostate_VA-PC-91-62
GSM1323669	Prostate_VA-PC-91-64
GSM1323670	Prostate_VA-PC-92-13
GSM1323671	Prostate_VA-PC-92-14
GSM1323672	Prostate_VA-PC-92-66
GSM1323673	Prostate_VA-PC-92-67
GSM1323674	Prostate_VA-PC-93-19
GSM1323675	Prostate_VA-PC-93-68
GSM1323676	Prostate_VA-PC-94-28
GSM1323677	Prostate_VA-PC-94-70
GSM1323678	Prostate_VA-PC-96-45
GSM1323679	Prostate_VA-PC-97-47
GSM1323680	Prostate_VA-PC-97-48
GSM1323681	Prostate_VA-PC-97-49
GSM1323682	Prostate_VA-PC-97-50
GSM1323683	Prostate_VA-PC-97-51
GSM1323684	Prostate_VA-PC-97-52
GSM1323685	Prostate_VA-PC-97-74
GSM1323686	Prostate_VA-PC-99-54
GSM1323687	Prostate_VA-PC-99-55
GSM1323688	Prostate_VA-PC-99-75
GSM1323689	Prostate_VA-PC-99-76
GSM1323690	Prostate_VA-PC-99-77
GSM1323691	Prostate_VA-PC-99-78
GSM1323692	Prostate_VA-PC-99-79
GSM1323693	Prostate_VA-PC-99-80
GSM1323694	Prostate_VA-PC-99-81
GSM1323695	Prostate_VA-PC-99-82
GSM1323696	Prostate_VA-PC-99-83
GSM1323697	Prostate_VA-PC-99-84
GSM1323698	Prostate_MCC-PT197
GSM1323699	Prostate_MCC-PT220
GSM1323700	Prostate_MCC-PT264
GSM1323701	Prostate_UTPC004-2
GSM1323702	Prostate_UTPC004
GSM1323703	Prostate_UTPC009-2
GSM1323704	Prostate_UTPC009
GSM1323705	Prostate_UTPC019-2
GSM1323706	Prostate_UTPC019
GSM1323707	Prostate_UTPC021-2
GSM1323708	Prostate_UTPC021
GSM1323709	Prostate_UTPC029-2
GSM1323710	Prostate_UTPC029
GSM1323711	Prostate_UTPC034-2
GSM1323712	Prostate_UTPC034
GSM1323713	Prostate_UTPC041
GSM1323714	Prostate_UTPC058
GSM1323715	Prostate_UTPC088
GSM1323716	Prostate_UTPC093
GSM1323717	Prostate_UTPC099
GSM1323718	Prostate_UTPC101
GSM1323719	Prostate_UTPC104
GSM1323720	Prostate_UTPC110
GSM1323721	Prostate_UTPC114
GSM1323722	Prostate_UTPC116
GSM1323723	Prostate_UTPC127
GSM1323724	Prostate_UTPC131
GSM1323725	Prostate_UTPC132
GSM1323726	Prostate_UTPC141
GSM1323727	Prostate_UTPC146
GSM1323728	Prostate_UTPC147
GSM1323729	Prostate_UTPC160
GSM1323730	Prostate_UTPC164
GSM1323731	Prostate_UTPC170
GSM1323732	Prostate_VA-PC-00-86
GSM1323733	Prostate_VA-PC-00-88
GSM1323734	Prostate_VA-PC-00-96
GSM1323735	Prostate_VA-PC-00-99
GSM1323736	Prostate_VA-PC-90-4
GSM1323737	Prostate_VA-PC-91-65
GSM1323738	Prostate_VA-PC-92-11
GSM1323739	Prostate_VA-PC-92-9
GSM1323740	Prostate_VA-PC-93-20
GSM1323741	Prostate_VA-PC-93-21
GSM1323742	Prostate_VA-PC-93-24
GSM1323743	Prostate_VA-PC-94-71
GSM1323744	Prostate_VA-PC-95-34
GSM1323745	Prostate_VA-PC-95-37
GSM1323746	Prostate_VA-PC-95-41
GSM1323747	Prostate_VA-PC-95-42
GSM1323748	Prostate_VA-PC-96-43
GSM1323749	Prostate_VA-PC-96-44
GSM1323750	Prostate_VA-PC-97-46
GSM1323751	Prostate_VA-PC-98-53
GSM1323752	Prostate_VA-PC-99-85

++++++++++
