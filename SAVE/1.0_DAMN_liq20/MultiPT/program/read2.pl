open (LOG, "<1.p.label.log");
while (<LOG>) { push @logg, [split]; }
my $i = 0;
open OUT, "+> OP17.txt";
open OUF, "+> OP5.txt";
open OUS, "+> OP6.txt";
open OUQ, "+> OP7.txt";#
open OUJ, "+> OP9.txt";#
open OUSHI,"+>OP10.txt";#
open OUSER,"+>OP12.txt";#
open OUSSA,"+>OP13.txt";#
open OUSSI,"+>OP14.txt";
open OUERS,"+>OP20.txt";
foreach my $a(@logg) {
		$pd = $a -> [0];
		#	$pd = ~ s/[\[\]]/ /g;
		print("$pd\n");
		#$pd = int($pd);
		my $index_i = $i;
		if (($pd eq "[17]")||($pd eq "[[17]")||($pd eq "[17]]")||($pd eq "[[17]]")) {
			$index_i = $index_i * 4;
			print OUT "$index_i ";
		}
		if (($pd eq "[5]")||($pd eq "[[5]")||($pd eq "[5]]")||($pd eq "[[5]]")) {
			$index_i = $index_i * 4;
			print OUF "$index_i ";
		}
		if (($pd eq "[6]")||($pd eq "[6]]")||($pd eq "[[6]")||($pd eq "[[6]]")) {
			$index_i = $index_i * 4;
			print OUS "$index_i ";
		}
		###
		if (($pd eq "[7]")||($pd eq "[7]]")||($pd eq "[[7]")||($pd eq "[[7]]")) {
			$index_i = $index_i * 4;
			print OUQ "$index_i ";
		}
		if (($pd eq "[9]")||($pd eq "[9]]")||($pd eq "[[9]")||($pd eq "[[9]]")) {
			$index_i = $index_i * 4;
			print OUJ "$index_i ";
		}
		if (($pd eq "[10]")||($pd eq "[10]]")||($pd eq "[[10]")||($pd eq "[[10]]")) {
			$index_i = $index_i * 4;
			print OUSHI "$index_i ";
		}
		if (($pd eq "[12]")||($pd eq "[12]]")||($pd eq "[[12]")||($pd eq "[[12]]")) {
			$index_i = $index_i * 4;
			print OUSER "$index_i ";
		}
		if (($pd eq "[13]")||($pd eq "[13]]")||($pd eq "[[13]")||($pd eq "[[13]]")) {
			$index_i = $index_i * 4;
			print OUSSA "$index_i ";
		}
		if (($pd eq "[14]")||($pd eq "[14]]")||($pd eq "[[14]")||($pd eq "[[14]]")) {
			$index_i = $index_i * 4;
			print OUSSI "$index_i ";
		}
		if (($pd eq "[20]")||($pd eq "[20]]")||($pd eq "[[20]")||($pd eq "[[20]]")) {
			$index_i = $index_i * 4;
			print OUERS "$index_i ";
		}
		$i++;
}
print OUT "\n";
print OUF "\n";
print OUS "\n";#
print OUQ "\n";
print OUJ "\n";
print OUSHI "\n";
print OUSER "\n";
print OUSSA "\n";
print OUSSI "\n";
print OUERS "\n";
close OUT;
close OUF;
close OUS;
close OUQ;
close OUJ ;
close OUSHI;
close OUSER;
close OUSSA;
close OUSSI;
close OUERS;
		#
