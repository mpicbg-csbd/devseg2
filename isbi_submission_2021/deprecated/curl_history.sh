
curl -v -k --ftp-ssl 'ftps://ftp.celltrackingchallenge.net:21/SW/CSB/single_cell_tracking.py' -u ctc148:Kq6Z9N -o my_sw.py
		stall ... Initializing NSS with certpath: sql:/etc/pki/nssdb

curl -v -k --ftp-ssl 'ftps://ftp.celltrackingchallenge.net/SW/CSB/single_cell_tracking.py' -u ctc148:Kq6Z9N -o my_sw.py
		run in inf loop, connects to port 990 ... 

curl -v 'ftps://ftp.celltrackingchallenge.net:21' -u ctc148:Kq6Z9N
		connect and stall ...  CAfile: /etc/pki/tls/certs/ca-bundle.crt

curl -v 'ftps://ftp.celltrackingchallenge.net:21' -u ctc148:Kq6Z9N
	fail

wget -v 'ftps://ctc148:Kq6Z9N@ftp.celltrackingchallenge.net:21' 
	fail
wget -v 'ftp://ctc148:Kq6Z9N@ftp.celltrackingchallenge.net:21' 
	fail

-bash-4.2$ curl -v --ssl-reqd 'ftp://ftp.celltrackingchallenge.net:21/SW/CSB/single_cell_tracking.py' -u ctc148:Kq6Z9N -o my_sw.py
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0* About to connect() to ftp.celltrackingchallenge.net port 21 (#0)
*   Trying 147.251.52.183...
* Connected to ftp.celltrackingchallenge.net (147.251.52.183) port 21 (#0)
< 220 FTP Server ready.
> AUTH SSL
< 234 AUTH SSL successful
* Initializing NSS with certpath: sql:/etc/pki/nssdb
*   CAfile: /etc/pki/tls/certs/ca-bundle.crt
  CApath: none
* SSL connection using TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
* Server certificate:
*       subject: CN=data.celltrackingchallenge.net
*       start date: May 07 06:04:43 2021 GMT
*       expire date: Aug 05 06:04:43 2021 GMT
*       common name: data.celltrackingchallenge.net
*       issuer: CN=R3,O=Let's Encrypt,C=US
> USER ctc148
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0< 331 Password required for ctc148
> PASS Kq6Z9N
< 230 User ctc148 logged in
> PBSZ 0
< 200 PBSZ 0 successful
> PROT P
< 200 Protection set to Private
> PWD
< 257 "/" is the current directory
* Entry path is '/'
> CWD SW
* ftp_perform ends with SECONDARY: 0
< 250 CWD command successful
> CWD CSB
< 250 CWD command successful
> EPSV
* Connect data stream passively
< 229 Entering Extended Passive Mode (|||58068|)
*   Trying 147.251.52.183...
* Connecting to 147.251.52.183 (147.251.52.183) port 58068
* Connected to ftp.celltrackingchallenge.net (147.251.52.183) port 21 (#0)
> TYPE I
< 200 Type set to I
> SIZE single_cell_tracking.py
< 213 10342
> RETR single_cell_tracking.py
< 150 Opening BINARY mode data connection for single_cell_tracking.py (10342 bytes)
* Maxdownload = -1
* Getting file with size: 10342
* Doing the SSL/TLS handshake on the data stream
*   CAfile: /etc/pki/tls/certs/ca-bundle.crt
  CApath: none
* SSL connection using TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
* Server certificate:
*       subject: CN=data.celltrackingchallenge.net
*       start date: May 07 06:04:43 2021 GMT
*       expire date: Aug 05 06:04:43 2021 GMT
*       common name: data.celltrackingchallenge.net
*       issuer: CN=R3,O=Let's Encrypt,C=US
{ [data not shown]
* transfer closed with 10342 bytes remaining to read
* Remembering we are in dir "SW/CSB/"
< 425 Unable to build data connection: Operation not permitted
* server did not report OK, got 425
  0 10342    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
* Connection #0 to host ftp.celltrackingchallenge.net left intact
curl: (18) transfer closed with 10342 bytes remaining to read
-bash-4.2$



-bash-4.2$ curl -v -P --ssl-reqd 'ftp://ftp.celltrackingchallenge.net:21/SW/CSB/single_cell_tracking.py' -u ctc148:Kq6Z9N -o my_sw.py
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0* About to connect() to ftp.celltrackingchallenge.net port 21 (#0)
*   Trying 147.251.52.183...
* Connected to ftp.celltrackingchallenge.net (147.251.52.183) port 21 (#0)
< 220 FTP Server ready.
> USER ctc148
< 550 SSL/TLS required on the control channel
* Access denied: 550
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
* Closing connection 0
curl: (67) Access denied: 550


curl -v --ssl-reqd 'ftp://ftp.celltrackingchallenge.net:21/SW/CSB/' -u ctc148:Kq6Z9N -T Makefile
