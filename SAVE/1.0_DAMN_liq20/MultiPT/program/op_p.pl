open (IN, "< index17.txt");
open (INT,"< panding.lammpstrj");
#open TEST, "+> test.log";
my @in = ();
while (<IN>) {push @in, [split];}
my $i = 0;
my @index_ = ();
foreach my $a(@in) {
	my $N = 0;
	for (my $j = 0; $j < 20000; $j ++) {
		my $u = $a -> [$j];
		if ((($u != 0)||($N == 0)) &&(length($u)>0) ) {
			$index_[$u] ++;
			#print "$u ($index_[$u]) ";
			#print "$u ";
			$N ++;
		}else{last;}
	}
	#print "\n";
	$i++;
}
##################
open (INY, "< index5.txt");
my @iny = ();
while (<INY>) {push @iny, [split];}
my $ii = 0;
my @index_2 = ();
foreach my $b(@iny){
	my $N = 0;
	for (my $j = 0; $j < 20000; $j ++) {
		my $u = $b -> [$j];
		if ((($u != 0) || ($N == 0)) &&(length($u)>0) ) {
			$index_2[$u] ++;
			$N ++;
		}else{last;}
	}
	$ii ++;
}
close INY;
###
open (INS, "< index6.txt");
my @ins = ();
while (<INS>) {push @ins, [split];}
my $jj = 0;
my @index_3 = ();
foreach my $b(@ins){
	my $N = 0;
	for (my $j = 0; $j < 20000; $j ++) {
		my $u = $b -> [$j];
		if ((($u != 0) || ($N == 0)) &&(length($u)>0) ) {
			$index_3[$u] ++;
			$N ++;
		}else{last;}
	}
	$jj ++;
}
close INS;
#####
open (IN7, "< index7.txt");
my @in7 = ();
while (<IN7>) {push @in7, [split];}
my $i7 = 0;
my @index_7 = ();
foreach my $c(@in7){
	my $N = 0;
	for (my $j = 0; $j<20000; $j++) {
		my $u = $c -> [$j];
		if ((($u != 0) || ($N == 0)) &&(length($u)>0) ) {
			$index_7[$u] ++;
			$N ++;
		}else{last;}
	}
	$i7++;
}
close IN7;
#####
open (IN9, "< index9.txt");
my @in9 = ();
while (<IN9>) {push @in9, [split];}
my $i9 = 0;
my @index_9 = ();
foreach my $d(@in9){
	my $N = 0;
	for (my $j = 0; $j<20000; $j++) {
		my $u = $d -> [$j];
		if ((($u != 0) || ($N == 0)) &&(length($u)>0) ) {
			$index_9[$u] ++;
			$N ++;
		}else{last;}
	}
	$i9++;
}
close IN9;
####
open (IN10, "< index10.txt");
my @in10 = ();
while (<IN10>) {push @in10, [split];}
my $i10 = 0;
my @index_10 = ();
foreach my $e(@in10){
	my $N = 0;
	for (my $j = 0; $j<20000; $j++) {
		my $u = $e -> [$j];
		if ((($u != 0) || ($N == 0)) &&(length($u)>0) ) {
			$index_10[$u] ++;
			$N ++;
		}else{last;}
	}
	$i10++;
}
close IN10;
####
open (IN12, "< index12.txt");
my @in12 = ();
while (<IN12>) {push @in12, [split];}
my $i12 = 0;
my @index_12 = ();
foreach my $f(@in12){
	my $N = 0;
	for (my $j = 0; $j<20000; $j++) {
		my $u = $f -> [$j];
		if ((($u != 0) || ($N == 0)) &&(length($u)>0) ) {
			$index_12[$u] ++;
			$N ++;
		}else{last;}
	}
	$i12++;
}
close IN12;
####
open (IN13, "< index13.txt");
my @in13 = ();
while (<IN13>) {push @in13, [split];}
my $i13 = 0;
my @index_13 = ();
foreach my $g(@in13){
	my $N = 0;
	for (my $j = 0; $j<20000; $j++) {
		my $u = $g -> [$j];
		if ((($u != 0) || ($N == 0)) &&(length($u)>0) ) {
			$index_13[$u] ++;
			$N ++;
		}else{last;}
	}
	$i13++;
}
close IN13;
####
open (IN14, "< index14.txt");
my @in14 = ();
while (<IN14>) {push @in14, [split];}
my $i14 = 0;
my @index_14 = ();
foreach my $h(@in14){
	my $N = 0;
	for (my $j = 0; $j<20000; $j++) {
		my $u = $h -> [$j];
		if ((($u != 0) || ($N == 0)) &&(length($u)>0) ) {
			$index_14[$u] ++;
			$N ++;
		}else{last;}
	}
	$i14++;
}
close IN14;
####
open (IN20, "< index20.txt");
my @in20 = ();
while (<IN20>) {push @in20, [split];}
my $i20 = 0;
my @index_20 = ();
foreach my $aa(@in20){
	my $N = 0;
	for (my $j = 0; $j<20000; $j++) {
		my $u = $aa -> [$j];
		if ((($u != 0) || ($N == 0))&&(length($u)>0)) {
			print $u;
			$index_20[$u] ++;
			print " [$index_20[$u]]";
			$N ++;
		}else{last;}
	}
	$i20++;
	print "\n";
}
close IN20;
my $test0= 0;
my $test0_length= length($test0);
print "$test0 : $test0_length\n";
####
print "Ntime $i $index_[4]\n";
##
open OUT1, "+> OP000.txt";
open OUT2, "+> OP015.txt";
open OUT3, "+> OP025.txt";
open OUT4, "+> OP050.txt";
open OUT5, "+> OP060.txt";
open OUT6, "+> OP080.txt";
open OUT17, "+> OP017.txt";
#my @intt = <INT>;
open OUT, "+> lammps_colour.lammpstrj";
@intt = ();
while(<INT>) {push @intt, [split]; }
my $linei = 0;
my $atomi = 0;
print "$i, $ii, $jj, $i7, $i9, $i10, $i12, $i13, $i14, $i20\n";
foreach my $fc(@intt) {
#$opi = 0; $opi < 20000; $opi ++) {
	if ($linei == 0) {
		my $lin1 = $fc -> [0];
		my $lin2 = $fc -> [1];
		print OUT "$lin1 $lin2\n";
	}
	if ($linei == 1) {
		print OUT "0\n";
	}
	if ($linei == 2) {
		print OUT "ITEM: NUMBER OF ATOMS\n";
	}
	if ($linei == 3) {
		my $lin1 = $fc -> [0];
		print OUT "$lin1\n";
	}
	if ($linei == 4) {
		print OUT "ITEM: BOX BOUNDS pp pp pp\n";
	}
	if ($linei == 5) {
		my $lin1 = $fc -> [0];
		my $lin2 = $fc -> [1];
		print OUT "$lin1 $lin2\n";
	}
	if ($linei == 6) {
		my $lin1 = $fc -> [0];
	        my $lin2 = $fc -> [1];
	        print OUT "$lin1 $lin2\n";
	}
	if ($linei == 7) {
		my $lin1 = $fc -> [0];
                my $lin2 = $fc -> [1];
                print OUT "$lin1 $lin2\n";
        }
	if ($linei == 8) {
		print OUT "ITEM: ATOMS id type xu yu zu ice17 ice1c ice1h ice2 ice3 ice4 ice5 ice6 ice7 ice20\n";
	}
	if ($linei > 8) {
		my $lin1 = $fc -> [0];
		my $lin2 = $fc -> [1];
		my $lin3 = $fc -> [2];#x
		my $lin4 = $fc -> [3];#y
		my $lin5 = $fc -> [4];#z
		my $colour1 = -1;
		my $colour2 = -1;
		my $colour3 = -1;
		my $colour4 = -1;
		my $colour5 = -1;
		my $colour6 = -1;
		my $colour7 = -1;
		my $colour8 = -1;
		my $colour9 = -1;
		my $colour10 = -1;
		if ($atomi % 4 ==0) {
			my $opi = int($atomi);
			if ($i == 0) {$colour1 = 0;}else{
				$colour1 = $index_[$opi]/$i;
				#print "$index_[$opi],";
				if ((!$colour1)) {$colour1=0;}#print"[0]";}
			}
			if ($ii== 0) {$colour2 = 0;}
			else{
				$colour2 = $index_2[$opi]/$ii;
				if ((!$colour2)) {$colour2=0;}
			}
			if ($jj== 0) {$colour3 = 0;}
			else{
				$colour3 = $index_3[$opi]/$jj;
				if ((!$colour3)) {$colour3=0;}
			}
			if ($i7== 0) {$colour4 = 0;}
			else{
				$colour4 = $index_7[$opi]/$i7;
				if ((!$colour4)) {$colour4=0;}
			}
			if ($i9== 0) {$colour5 = 0;}
			else{
				$colour5 = $index_9[$opi]/$i9;
				if ((!$colour5)) {$colour5=0;}
			}
			if ($i10==0) {$colour6 = 0;}
			else{
				$colour6 = $index_10[$opi]/$i10;
				if ((!$colour6)) {$colour6=0;}
			}
			if ($i12==0) {$colour7 = 0;}
			else{
				$colour7 = $index_12[$opi]/$i12;
				if ((!$colour7)) {$colour7=0;}
			}
			if ($i13==0) {$colour8 = 0;}
			else{
				$colour8 = $index_13[$opi]/$i13;
				if ((!$colour8)) {$colour8=0;}
			}
			if ($i14==0) {$colour9 = 0;}
			else{
				$colour9 = $index_14[$opi]/$i14;
				if ((!$colour9)) {$colour9=0;}
			}
			if ($i20==0) {$colour10= 0;}
			else{
				$colour10= $index_20[$opi]/$i20;
				if ((!$colour10)) {$colour10=0;}
			}
			#if ($linei > 8) {}
			if (($colour2+$colour3)>0.1) {
				print OUT1 " $opi";
			}
			if (($colour2+$colour3)>0.15) {
				print OUT2 " $opi";
			}
			if (($colour2+$colour3)>0.20) {
				print OUT3 " $opi";
			}
			if (($colour2+$colour3)>0.25) {
				print OUT4 " $opi";
			}
			if (($colour2+$colour3)>0.30) {
				print OUT5 " $opi";
			}
			if (($colour2+$colour3)>0.50) {
				print OUT6 " $opi";
			}
			if ($colour1>0) {
				print OUT17 " $opi";
			}
			print OUT "$lin1 $lin2 $lin3 $lin4 $lin5 $colour1 $colour2 $colour3 $colour4 $colour5 $colour6 $colour7 $colour8 $colour9 $colour10\n";
		}else{
			print OUT "$lin1 $lin2 $lin3 $lin4 $lin5 0 0 0 0 0 0 0 0 0 0\n"
		}
		$atomi ++;
	}
	$linei ++;
}
print OUT1 "\n";
print OUT2 "\n";
print OUT3 "\n";
print OUT4 "\n";
print OUT5 "\n";
print OUT6 "\n";
print OUT17 "\n";
close OUT1;
close OUT2;
close OUT3;
close OUT4;
close OUT5;
close OUT6;
close OUT17;

close IN;
close INT;
close OUT;
#close TEST;
#close OUTD;
