# 1.0_DAMN_liq20/MultiPT/P5000_T200
my $cal_filename = 'only1';  #'P1_T230'
$cal_filename = $ARGV[0]
#
sub read_OP_distribution {
	open IN, "< ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/Uni-OP.txt";
	my @in = ();
	my $num = 0;
	my @size = ();
	while (<IN>) {push @in, [split];}
	foreach my $a (@in) {
		$size[$num] = $a -> [0];
		$num ++;
	}
	close IN;
	return ($num, \@size);
}

sub Save_nucleus {
	($file1, $file2, $save_n) = @_;
	system ("cp $file1 $file2");
	system ("cd ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/. && perl F.pl");
	system ("cd ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/. && perl liantong_all.pl");
	##
	my ($a1, $a2) = read_OP_distribution();
	open WRITE, ">> $save_n";
	print WRITE "$a1 |";
	foreach my $b(@$a2) {
		print WRITE " $b";
	}
	print WRITE "\n";
	close WRITE;
}

open OUT, "+> ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/index17.txt";
open OUF, "+> ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/index5.txt";
open OUS, "+> ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/index6.txt";
open OU7, "+> ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/index7.txt";#
open OU9, "+> ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/index9.txt";#
open OU10,"+> ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/index10.txt";#
open OU12,"+> ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/index12.txt";#
open OU13,"+> ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/index13.txt";#
open OU14,"+> ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/index14.txt";
open OU20,"+> ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/index20.txt";
for (my $i = 0; $i< 100; $i++) {
	system("python3 Uni-OP_v0.2_testing.py");
	system("cd ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/. && perl read2.pl");
	open(RE, "< ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/OP17.txt");
	my @re = <RE>;
	print OUT "@re";
	close RE;
	open(RT, "< ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/OP5.txt");
	my @rt = <RT>;
	print OUF "@rt";
	close RT;
	open(ST, "< ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/OP6.txt");
	my @st = <ST>;
	print OUS "@st";
	close ST;
	###
	open(RE, "< ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/OP7.txt");	
	@re = ();
	@re = <RE>;
	print OU7 "@re";
	close RE;
	###
	open(RE, "< ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/OP9.txt");
	@re = ();
	@re = <RE>;
	print OU9 "@re";
	close RE;
	###
	open(RE, "< ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/OP10.txt");
	@re = ();
	@re = <RE>;
	print OU10 "@re";
	close RE;
	###
	open(RE, "< ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/OP12.txt");
	@re = ();
	@re = <RE>;
	print OU12 "@re";
	close RE;
	###
	open(RE, "< ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/OP13.txt");
	@re = ();
	@re = <RE>;
	print OU13 "@re";
	close RE;
	###
	open(RE, "< ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/OP14.txt");
	@re = ();
	@re = <RE>;
	print OU14 "@re";
	close RE;
	###
	open(RE, "< ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/OP20.txt");
	@re = ();
	@re = <RE>;
	print OU20 "@re";
	close RE;
	###
	system("cd ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/. && rm *log OP17.txt OP5.txt OP6.txt OP7.txt OP9.txt OP10.txt OP12.txt OP13.txt OP14.txt OP20.txt");
	###
}
close OUT;
close OUF;
close OUS;
close OU7;
close OU9;
close OU10;
close OU12;
close OU13;
close OU14;
close OU20;
##
system("cp ../SAVE/MultiPT/$cal_filename/56.panding.gro ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/panding.gro");
system("cd ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/. && ./cj");
system("cd ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/. && perl op_p.pl");
##
#system("cp OP000.txt OP");
&Save_nucleus("../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/OP000.txt","../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/OP.txt","../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/UniOP_000.txt");
&Save_nucleus("../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/OP015.txt","../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/OP.txt","../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/UniOP_015.txt");
&Save_nucleus("../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/OP025.txt","../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/OP.txt","../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/UniOP_025.txt");
&Save_nucleus("../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/OP050.txt","../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/OP.txt","../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/UniOP_050.txt");
&Save_nucleus("../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/OP060.txt","../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/OP.txt","../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/UniOP_060.txt");
&Save_nucleus("../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/OP080.txt","../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/OP.txt","../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/UniOP_080.txt");
&Save_nucleus("../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/OP017.txt","../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/OP.txt","../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/UniOP_17.txt");
system("cd ../SAVE/1.0_DAMN_liq20/MultiPT/$cal_filename/. && rm OP017.txt index17.txt index5.txt index6.txt index7.txt index9.txt index10.txt index12.txt index13.txt index14.txt index20.txt OP.txt OP000.txt OP015.txt OP025.txt OP050.txt OP060.txt OP080.txt Uni-OP.txt panding.gro F.txt cj1.txt");
#system("perl F.pl");
#system("perl liantong_all.pl");
###
