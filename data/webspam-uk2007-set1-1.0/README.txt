
                      WEBSPAM-UK2007 LABELS
         http://www.yr-bcn.es/webspam/datasets/uk2007/

VERSION HISTORY
Version 1.0 - Released 2008-01-15

AUTHORS
Several contributors, see CREDITS.txt

LICENSE OF THESE LABELS
Creative Commons Attribution-Noncommercial-Share Alike 3.0
http://creativecommons.org/licenses/by-nc-sa/3.0/
You can download and use the labels for research in any institution
public or private. The "nc-sa" (non-commercial, share-alike) rule
applies if you want to redistribute the labels publicly.  

DESCRIPTION
These files contain a set of assessments done by a group of volunteers
over a set of hosts. For the purpose of the Web Spam Challenge 2008,
the labels are being released in two sets. SET1, containing roughly 2/3
of the assessed hosts will be given for training, while SET2 containing
the remaining 1/3, will be held for testing.  

The partition of the hosts into SET1 and SET2 was done at random based on the
third-level domain of the hosts, i.e., hosts www1.example.co.uk and
www2.example.co.uk belong to the same partition.

FILE: WEBSPAM-UK2007-hostnames.txt
This file contains the hostid to hostname mapping.

 hostid   -- a unique number from 0 to 114,528 identifying each host
             in the collection
 hostname -- the hostname and port if it is not the standard HTTP port 80

-------sample---------
0 www.exampleA.co.uk
1 www.exampleB.co.uk
2 www.exampleC.co.uk
...
----------------------

FILE: WEBSPAM-UK2007-SET1-labels.txt
This file contains the actual labels.

 hostid      -- the hostid as specified in the hostnames.txt file
 label       -- one of {nonspam, spam, undecided} depending on the spamicity
 spamicity   -- the average of the assessments considering
                nonspam=0, spam=1, borderline=0.5, and not counting 'unknown'
                labels. If no valid label is present, a '-' sign is written
 assessments -- a comma-separated list of individual assessments, considering
                the most up-to-date assessment for each assessor.
                The assessments are N=nonspam, S=spam, B=borderline, U=unknown

-------sample---------
5 normal 0.00000 j1:N,j2:N
8 normal 0.33333 j14:N,j17:S,j7:N
12 spam 1.00000 j18:U,j4:S
17 undecided - j13:U,j20:U
21 undecided 0.50000 j15:N,j16:S,j22:U
----------------------

FILE: WEBSPAM-UK2007-SET1-raw-assessments.txt
This file contains a log of the assessment process.

The assessment took place in two phases. During the first phase, the 
assessors had access only to the contents of the collection and the
list of linked hosts for each assessed host. During the second phase,
the assessors had access also to the labels provided by other assessors,
and were asked to review their assessments, particularly when they disagreed
with the others.

 hostid      -- the hostid as specified in the hostnames.txt file
 assessor    -- the id of the assessor
 label       -- the label given, or '-' if this is a VIEW
 timestamp   -- the timestamp at which the event happened
 period      -- the type of assessment; INITIAL=first phase, UPDATED=updated
                label during the first phase REVISED=revised label during
                the second phase, VIEW=view the host. Only the first
                view is recorded, not individual page views of the assessor.

-------sample---------
5 j1 - 1041070108158 VIEW
5 j1 borderline 1041070108160 INITIAL
5 j1 - 1041070108172 VIEW
5 j1 spam 1041070108173 UPDATED
12 j4 - 1043090108180 VIEW
12 j4 normal 1043090108190 INITIAL
12 j4 - 1043090108200 VIEW
12 j4 spam 1044090108210 REVISED
-----------------------

STATISTICS
This is the prevalence of each label in SET1 and SET2.

SET1, given for training in the Web Spam Challenge 2008:
   3776 nonspam
    222 spam
    277 undecided

SET2, held for testing in the Web Spam Challenge 2008:
   1933 nonspam
    122 spam
    149 undecided

